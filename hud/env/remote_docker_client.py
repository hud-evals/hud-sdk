from __future__ import annotations

import logging
from base64 import b64decode, b64encode
from typing import Any

from hud.env.docker_client import DockerClient
from hud.server import make_request
from hud.settings import settings
from hud.types import EnvironmentStatus
from hud.utils import ExecuteResult
from hud.utils.common import get_gym_id

logger = logging.getLogger("hud.env.remote_env_client")


class RemoteDockerClient(DockerClient):
    """
    Remote environment client implementation.

    Uses the HUD API to manage a remote environment.
    """

    @classmethod
    async def create(
        cls,
        dockerfile: str,
        *,
        job_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[RemoteDockerClient, dict[str, Any]]:
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

        logger.info("Creating remote environment")

        true_gym_id = await get_gym_id("local-docker")

        # augment metadata with dockerfile
        if "environment_config" not in metadata:
            metadata["environment_config"] = {}

        metadata["environment_config"]["dockerfile"] = dockerfile

        # Create a new environment via the HUD API
        response = await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/create_environment",
            json={
                # still named run_id for backwards compatibility
                "run_id": job_id,
                "metadata": metadata,
                "gym_id": true_gym_id,
                "task_id": task_id,
            },
            api_key=settings.api_key,
        )

        # Get the environment ID from the response
        env_id = response.get("id")
        if not env_id:
            raise ValueError("Failed to create remote environment: No ID returned")

        # Create the controller instance
        controller = cls(env_id)

        build_metadata = response.get("metadata", {})

        return controller, build_metadata

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
            exit_code=data["exit_code"],
        )

    async def get_archive(self, path: str) -> bytes:
        """
        Get an archive of a path from the environment.
        May not be supported for all environments.

        Args:
            path: Path in the environment to archive

        Returns:
            bytes: Content of the file or archive
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/environments/{self.env_id}/get_archive",
            json={"path": path},
            api_key=settings.api_key,
        )

        # Return the content decoded from base64
        return b64decode(data["content"])

    async def put_archive(self, path: str, data: bytes) -> bool:
        """
        Put an archive of data at a path in the environment.
        May not be supported for all environments.

        Args:
            path: Path in the environment to extract the archive to
            data: Bytes of the data to send

        Returns:
            bool: True if successful
        """
        await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/environments/{self.env_id}/put_archive",
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
        await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/environments/{self.env_id}/close",
            api_key=settings.api_key,
        )
