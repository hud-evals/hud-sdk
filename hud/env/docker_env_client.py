import uuid
from aiodocker.stream import Stream
from aiohttp import ClientTimeout
from typing import Optional
from hud.env import EnvClient
from hud.utils import ExecuteResult
import io
import aiodocker
from hud.env.env_client import EnvironmentStatus


class DockerEnvClient(EnvClient):
    """
    Docker-based environment client implementation.
    """
    
    @classmethod
    async def create(cls, dockerfile: str) -> 'DockerEnvClient':
        """
        Creates a Docker environment client from a dockerfile.
        
        Args:
            dockerfile: The dockerfile content to build the Docker image
            
        Returns:
            DockerEnvClient: An instance of the Docker environment client
        """
        # Create a unique image tag
        image_tag = f"hud-env-{uuid.uuid4().hex[:8]}"
        
        # Initialize Docker client
        docker_client = aiodocker.Docker()
        
        # Create fileobj for the Dockerfile
        dockerfile_fileobj = io.BytesIO(dockerfile.encode("utf-8"))

        # Build the image
        print(f"Building Docker image {image_tag}...")
        build_stream = await docker_client.images.build(
            fileobj=dockerfile_fileobj,
            tag=image_tag,
            rm=True,
            pull=True,
            forcerm=True,
        )
        
        # Print build output
        for chunk in build_stream:
            if "stream" in chunk:
                print(chunk["stream"], end="")
        
        # Create and start the container
        print(f"Starting Docker container from image {image_tag}...")
        container_config = {
            "Image": image_tag,
            "Tty": True,
            "OpenStdin": True,
            "Cmd": ["/bin/bash"],
            "HostConfig": {
                "AutoRemove": True,
            }
        }
        
        container = await docker_client.containers.create(
            config=container_config
        )
        await container.start()
        
        # Return the controller instance
        return cls(docker_client, container.id)
    
    def __init__(self, docker_conn: aiodocker.Docker, container_id: str):
        """
        Initialize the DockerEnvClient.
        
        Args:
            docker_conn: Docker client connection
            container_id: ID of the Docker container to control
        """
        super().__init__()
        
        # Store container ID instead of container object
        self.container_id = container_id
        
        # Docker client will be initialized when needed
        self._docker = docker_conn
        
    async def _get_container(self):
        """Get the container object from aiodocker."""
        return await self._docker.containers.get(self.container_id)
    
    async def get_status(self) -> EnvironmentStatus:
        """
        Get the current status of the Docker environment.
        
        Returns:
            EnvironmentStatus: The current status of the environment
        """
        try:
            container = await self._get_container()
            container_data = await container.show()
            
            # Check the container state
            state = container_data.get("State", {})
            status = state.get("Status", "").lower()
            
            if status == "running":
                return EnvironmentStatus.RUNNING
            elif status == "created" or status == "starting":
                return EnvironmentStatus.INITIALIZING
            elif status in ["exited", "dead", "removing", "paused"]:
                return EnvironmentStatus.COMPLETED
            else:
                # Any other state is considered an error
                return EnvironmentStatus.ERROR
                
        except Exception:
            # If we can't connect to the container or there's any other error
            return EnvironmentStatus.ERROR
    
    async def execute(self, command: list[str], *, workdir: Optional[str] = None, timeout: Optional[float] = None) -> ExecuteResult:
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

    async def close(self) -> None:
        """
        Close the Docker environment by stopping and removing the container.
        """
        try:
            container = await self._get_container()
            await container.stop()
            await container.delete()
        except Exception as e:
            # Log the error but don't raise it since this is cleanup
            print(f"Error closing Docker environment: {e}")