from pathlib import Path
from aiodocker.stream import Stream
from aiohttp import ClientTimeout
from pydantic import BaseModel
import abc
from typing import Literal, Optional, Union
from hud.utils import ExecuteResult
from hud.utils.common import directory_to_tar_bytes
import io
import aiodocker
from base64 import b64decode, b64encode
from hud.server import make_request
from hud.settings import settings

# Env controller specs:
# These represent the python package that handles commands like "step" and "reset"
# and the environment itself. They are responsible for managing the environment's lifecycle.

class DevEnvControllerSpec(BaseModel):
    """Environment controller for development environments with a local path."""
    type: Literal["dev"] = "dev"
    path: str


class RemoteEnvControllerSpec(BaseModel):
    """Environment controller for remote environments identified by an id."""
    type: Literal["remote"] = "remote"
    id: str


# Environment specifications:
# These repsent the environment as a whole, including both the controller and the environment type (eg, what os, which services are running)

class PrivateEnvSpec(BaseModel):
    """Private environment specification identified by an id."""
    type: Literal["private"] = "private"
    id: str


class PublicEnvSpec(BaseModel):
    """Public environment specification with a dockerfile and controller."""
    type: Literal["public"] = "public"
    dockerfile: str
    controller_spec: Union[DevEnvControllerSpec, RemoteEnvControllerSpec]

# Environment Controllers

class EnvController(BaseModel, abc.ABC):
    """Base class for environment controllers."""
    
    @abc.abstractmethod
    async def needs_update(self) -> bool:
        """Check if the controller needs an update."""
        pass
    
    @abc.abstractmethod
    async def update(self) -> None:
        """Update the environment controller."""
        pass
    
    @abc.abstractmethod
    async def execute(self, command: list[str], *, workdir: Optional[str]) -> ExecuteResult:
        """Execute a command in the environment."""
        pass
    
    @abc.abstractmethod
    async def get_archive(self, path: str) -> bytes:
        """Get an archive of a path from the environment."""
        pass
    
    @abc.abstractmethod
    async def put_archive(self, path: str, data: bytes) -> bool:
        """Put an archive of data at a path in the environment."""
        pass

class DevEnvController(EnvController):
    """
    Development environment controller.
    
    Uses docker to manage a local environment.
    
    The docker container must already be running 
    
    """
    
    def __init__(self, container_id: str, source_path: str):
        """
        Initialize the DevEnvController.
        
        Args:
            container_id: ID of the Docker container to control
            source_path: Path to the source code to copy to the container
        """
        # Store container ID instead of container object
        self.container_id = container_id
        self.source_path = Path(source_path)
        assert self.source_path.exists(), f"Source path {self.source_path} does not exist"
        assert self.source_path.is_dir(), f"Source path {self.source_path} is not a directory"
        
        # the last known pyproject.toml
        self.last_pyproject_toml_str = None
        
        # Docker client will be initialized when needed
        self._docker = None
        
    async def _get_container(self):
        """Get the container object from aiodocker."""
        if self._docker is None:
            self._docker = aiodocker.Docker()
        return await self._docker.containers.get(self.container_id)
    
    async def needs_update(self) -> bool:
        """
        Check if the environment needs an update by comparing the current pyproject.toml
        with the previously saved version.
        
        Returns:
            bool: True if the environment needs an update, False otherwise.
        """
        pyproject_path = self.source_path / "pyproject.toml"
        
        # If pyproject.toml doesn't exist, we can't determine if an update is needed
        if not pyproject_path.exists():
            return False
        
        # Read the current content of pyproject.toml
        current_pyproject_content = pyproject_path.read_text()
        
        # If we don't have a saved version or if it's different from the current one
        if self.last_pyproject_toml_str is None or self.last_pyproject_toml_str != current_pyproject_content:
            return True
            
        return False
    
    async def update(self) -> None:
        """
        Update the environment by copying the source code to the container.
        Saves the current pyproject.toml content for future comparison.
        
        Raises:
            FileNotFoundError: If pyproject.toml doesn't exist in the source path.
        """
        # Check if pyproject.toml exists
        pyproject_path = self.source_path / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found in {self.source_path}")
        
        # Save current pyproject.toml content
        self.last_pyproject_toml_str = pyproject_path.read_text()
        
        # Create tar archive of the source code and send it to the container
        tar_bytes = directory_to_tar_bytes(self.source_path)
        await self.execute(["mkdir", "-p", "/root/controller"], workdir=None, timeout=5)        
        await self.put_archive("/root/controller", tar_bytes)
    
    async def execute(self, command: list[str], *, workdir: Optional[str], timeout: float) -> ExecuteResult:
        """
        Execute a command in the container.
        
        Args:
            command: Command to execute
            workdir: Working directory for the command
            
        Returns:
            ExecuteResult: Result of the command execution
        """
        container = await self._get_container()
        

        exec_result = await container.exec(
            cmd=command,
            workdir=workdir,
        )
        output: Stream = exec_result.start(
            timeout=ClientTimeout(timeout),
            detach=False
        )
        
        stdout_data = bytearray()
        stderr_data = bytearray()
        
        while True:
            message = await output.read_out()
            if message is None:
                break
            if message.stream == 1:  # stdout
                stdout_data.extend(message.data)
            elif message.stream == 2:  # stderr
                stderr_data.extend(message.data)

        
        return ExecuteResult(
            stdout=bytes(stdout_data),
            stderr=bytes(stderr_data),
            # TODO: Get the exit code from the output
            exit_code=0
        )
    
    async def get_archive(self, path: str) -> bytes:
        """
        Get an archive of a path from the container.
        
        Args:
            path: Path in the container to archive
            
        Returns:
            bytes: Tar archive containing the path contents
        """
        container = await self._get_container()
        
        tarfile = await container.get_archive(path)
        # we know tarfile has fileobj BytesIO
        # read the tarfile into a bytes object
        fileobj = tarfile.fileobj
        assert isinstance(fileobj, io.BytesIO), "fileobj is not a BytesIO object"
        return fileobj.getvalue()
    
    async def put_archive(self, path: str, data: bytes) -> None:
        """
        Put an archive of data at a path in the container.
        
        Args:
            path: Path in the container to extract the archive to
            data: Bytes of the tar archive to extract
            
        Returns:
            bool: True if successful
        """
        container = await self._get_container()
        
        # Convert bytes to a file-like object for aiodocker
        file_obj = io.BytesIO(data)
        await container.put_archive(path=path, data=file_obj)


class RemoteEnvController(EnvController):
    """
    Remote environment controller.
    
    Uses the HUD API to manage a remote environment.
    """
    
    def __init__(self, env_id: str):
        """
        Initialize the RemoteEnvController.
        
        Args:
            env_id: ID of the remote environment to control
        """
        self.env_id = env_id
    
    async def needs_update(self) -> bool:
        """
        Check if the controller needs an update.
        
        For remote controllers, this always returns False as the remote environment
        is managed by the server.
        
        Returns:
            bool: Always False for remote environments
        """
        return False
    
    async def update(self) -> None:
        """
        Update the environment controller.
        
        For remote controllers, this is a no-op as the remote environment
        is managed by the server.
        """
        # No-op for remote environments
        pass
    
    async def execute(self, command: list[str], *, workdir: Optional[str]) -> ExecuteResult:
        """
        Execute a command in the environment.
        
        Args:
            command: Command to execute
            workdir: Working directory for the command (ignored for remote environments)
            
        Returns:
            ExecuteResult: Result of the command execution
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.env_id}/execute",
            json=command,
            api_key=settings.api_key,
        )
        
        return ExecuteResult(
            stdout=b64decode(data["stdout"]),
            stderr=b64decode(data["stderr"]),
            exit_code=data["exit_code"]
        )
    
    async def get_archive(self, path: str) -> bytes:
        """
        Get an archive of a path from the environment.
        
        Args:
            path: Path in the environment to archive
            
        Returns:
            bytes: Content of the file or archive
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.env_id}/copy_from",
            json={"path": path},
            api_key=settings.api_key,
        )
        
        # Return the content decoded from base64
        return b64decode(data["content"])
    
    async def put_archive(self, path: str, data: bytes) -> bool:
        """
        Put an archive of data at a path in the environment.
        
        Args:
            path: Path in the environment to extract the archive to
            data: Bytes of the data to send
            
        Returns:
            bool: True if successful
        """
        await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.env_id}/copy_to",
            json={
                "path": path,
                "content": b64encode(data).decode("utf-8"),
            },
            api_key=settings.api_key,
        )
        
        return True
