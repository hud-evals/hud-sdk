import asyncio
from typing import Optional
from hud.env.env_client import EnvClient
from hud.utils import ExecuteResult
from base64 import b64decode, b64encode
from hud.server import make_request
from hud.settings import settings
from hud.types import EnvironmentStatus


class RemoteEnvClient(EnvClient):
    """
    Remote environment client implementation.
    
    Uses the HUD API to manage a remote environment.
    """
    
    @classmethod
    async def create(cls, dockerfile: str) -> 'RemoteEnvClient':
        """
        Creates a remote environment client from a dockerfile.
        
        Args:
            dockerfile: The dockerfile content to build the environment
            
        Returns:
            RemoteEnvClient: An instance of the remote environment client
        """
        # Create a new environment via the HUD API
        response = await make_request(
            method="POST",
            url=f"{settings.base_url}/environments",
            json={
                "dockerfile": dockerfile,
            },
            api_key=settings.api_key,
        )
        
        # Get the environment ID from the response
        env_id = response.get("id")
        if not env_id:
            raise ValueError("Failed to create remote environment: No ID returned")
        
        # Create the controller instance
        controller = cls(env_id)
        
        # Wait for the environment to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                # Check if the environment is ready by executing a simple command
                result = await controller.execute(["echo", "ready"], workdir=None)
                if result["stdout"].strip() == b"ready":
                    break
            except Exception as e:
                if i == max_retries - 1:
                    raise TimeoutError(f"Environment creation timed out after {max_retries} retries") from e
                await asyncio.sleep(2)  # Wait before retrying
        
        return controller
    
    def __init__(self, env_id: str):
        """
        Initialize the RemoteEnvClient.
        
        Args:
            env_id: ID of the remote environment to control
        """
        super().__init__()
        self.set_env_id(env_id)
    
    async def get_status(self) -> EnvironmentStatus:
        """
        Get the current status of the remote environment.
        
        Returns:
            EnvironmentStatus: The current status of the environment
        """
        try:
            response = await make_request(
                method="GET",
                url=f"{settings.base_url}/environments/{self.env_id}",
                api_key=settings.api_key,
            )
            
            # Map the API status to our EnvironmentStatus enum
            status = response.get("status", "").lower()
            
            if status == "running":
                return EnvironmentStatus.RUNNING
            elif status == "initializing" or status == "pending":
                return EnvironmentStatus.INITIALIZING
            elif status == "completed" or status == "terminated":
                return EnvironmentStatus.COMPLETED
            else:
                # Any other status is considered an error
                return EnvironmentStatus.ERROR
                
        except Exception:
            # If we can't connect to the API or there's any other error
            return EnvironmentStatus.ERROR
    
    async def execute(self, command: list[str], *, workdir: Optional[str] = None, timeout: Optional[float] = None) -> ExecuteResult:
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

    async def close(self) -> None:
        """
        Close the remote environment by making a request to the server.
        """
        try:
            await make_request(
                method="POST",
                url=f"{settings.base_url}/environments/{self.env_id}/close",
                api_key=settings.api_key,
            )
        except Exception as e:
            # Log the error but don't raise it since this is cleanup
            print(f"Error closing remote environment: {e}")
