from __future__ import annotations

import logging
from base64 import b64decode
from typing import TYPE_CHECKING, Any

from hud.env.client import Client
from hud.server import make_request
from hud.settings import settings
from hud.types import EnvironmentStatus
from hud.utils import ExecuteResult

if TYPE_CHECKING:
    from hud.utils.config import HudStyleConfig

logger = logging.getLogger("hud.env.remote_env_client")

class RemoteClient(Client):
    """
    Remote environment client implementation.
    
    Uses the HUD API to manage a remote environment.
    """
    
    @classmethod
    async def create(
        cls,
        *,
        gym_id: str | None = None,
        job_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[RemoteClient, dict[str, Any]]:
        """
        Creates a remote environment client from a dockerfile or gym_id.
        
        Args:
            dockerfile: The dockerfile content to build the environment
            gym_id: The gym_id of the environment to create
            metadata: Metadata to associate with the environment
            
        Returns:
            RemoteClient: An instance of the remote environment client
        """

        # Validate arguments
        if metadata is None:
            metadata = {}

        
        request_data = {
            # still named run_id for backwards compatibility
            "run_id": job_id,
            "metadata": metadata,
            "gym_id": gym_id,
            "task_id": task_id,
        }

        # Create a new environment via the HUD API
        response = await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/create_environment",
            json=request_data,
            api_key=settings.api_key,
        )
        
        # Get the environment ID from the response
        env_id = response.get("id")
        if not env_id:
            raise ValueError("Failed to create remote environment: No ID returned")
        
        # Create the controller instance
        controller = cls(env_id)
        
        build_data = response.get("metadata", {})
        
        if response.get("readme"):
            logger.info("[HUD] %s gym created, see how to use it at %s", gym_id,
                        response.get("readme"))
        
        return controller, build_data

    def __init__(self, env_id: str) -> None:
        """
        Initialize the RemoteClient.
        
        Args:
            env_id: ID of the remote environment to control
        """
        super().__init__()
        self._env_id = env_id
        
    @property
    def env_id(self) -> str:
        """The ID of the remote environment."""
        return self._env_id

    async def get_status(self) -> EnvironmentStatus:
        """
        Get the current status of the remote environment.
        
        Returns:
            EnvironmentStatus: The current status of the environment
        """
        try:
            response = await make_request(
                method="GET",
                url=f"{settings.base_url}/v2/environments/{self.env_id}/state",
                api_key=settings.api_key,
            )
            logger.debug("Environment status response: %s", response)

            status = response.get("state", "").lower()
            
            if status == "running":
                return EnvironmentStatus.RUNNING
            elif status == "initializing" or status == "pending":
                return EnvironmentStatus.INITIALIZING
            elif status == "completed" or status == "terminated":
                return EnvironmentStatus.COMPLETED
            else:
                # Any other status is considered an error
                logger.warning("Abnormal environment status response: %s", response)
                return EnvironmentStatus.ERROR
                
        except Exception:
            # If we can't connect to the API or there's any other error
            logger.info("(potentially transient) Error getting environment status")
            return EnvironmentStatus.ERROR
    
    async def execute(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> ExecuteResult:
        """
        Execute a command in the environment.
        No-op in some environments (like browser use).
        
        Args:
            command: Command to execute
            workdir: Working directory for the command (ignored for remote environments)
            
        Returns:
            ExecuteResult: Result of the command execution
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/environments/{self.env_id}/execute",
            json={
               "command": command,
               "workdir": workdir,
               "timeout": timeout,
            },
            api_key=settings.api_key,
        )
        
        return ExecuteResult(
            stdout=b64decode(data["stdout"]),
            stderr=b64decode(data["stderr"]),
            exit_code=data["exit_code"]
        )

    
    async def invoke(self, config: HudStyleConfig) -> tuple[Any, bytes, bytes]:
        """
        Invoke a function in the environment.
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/environments/{self.env_id}/invoke",
            json=config.model_dump(),
            api_key=settings.api_key,
        )
        
        return data["result"], b64decode(data["stdout"]), b64decode(data["stderr"])


    async def close(self) -> None:
        """
        Close the remote environment by making a request to the server.
        """
        await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/environments/{self.env_id}/close",
            api_key=settings.api_key,
        )
