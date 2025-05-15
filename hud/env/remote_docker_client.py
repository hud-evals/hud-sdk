from __future__ import annotations

import logging
from base64 import b64decode, b64encode
from typing import TYPE_CHECKING, Any

import httpx

from hud.env.docker_client import DockerClient
from hud.exceptions import HudResponseError
from hud.server import make_request
from hud.settings import settings
from hud.types import EnvironmentStatus
from hud.utils import ExecuteResult
from hud.utils.common import directory_to_zip_bytes, get_gym_id

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("hud.env.remote_env_client")


async def upload_bytes_to_presigned_url(presigned_url: str, data_bytes: bytes) -> None:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(presigned_url, content=data_bytes)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.exception("Failed to upload to presigned URL")
        raise HudResponseError(message=f"Failed to upload to presigned URL: {e}") from e
    except httpx.RequestError as e:
        logger.exception("Network error uploading to presigned URL")
        raise HudResponseError(message=f"Network error uploading to presigned URL: {e}") from e


class RemoteDockerClient(DockerClient):
    """
    Remote environment client implementation.

    Uses the HUD API to manage a remote environment.
    """

    @classmethod
    async def build_image(cls, build_context: Path) -> tuple[str, dict[str, Any]]:
        """
        Build an image from a build context.
        """
        # create the presigned url by making a POST request to /v2/builds
        logger.info("Creating build")
        response = await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/builds",
            api_key=settings.api_key,
        )
        logger.info("Build created")
        presigned_url = response["presigned_url"]

        # List files in the build context
        files = list(build_context.glob("**/*"))
        logger.info("Found %d files in build context %s", len(files), build_context)

        if len(files) == 0:
            raise HudResponseError(message="Build context is empty")

        # zip the build context
        logger.info("Zipping build context")
        zip_bytes = directory_to_zip_bytes(build_context)
        logger.info("Created zip archive of size %d kb", len(zip_bytes) // 1024)
        # upload the zip bytes to the presigned url
        logger.info("Uploading build context")
        await upload_bytes_to_presigned_url(presigned_url, zip_bytes)
        logger.info("Build context uploaded")

        # start the build and return uri and logs
        logger.info("Starting build")
        response = await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/builds/{response['id']}/start",
            api_key=settings.api_key,
        )
        logger.info("Build completed")

        return response["uri"], {"logs": response["logs"]}

    @classmethod
    async def create(
        cls,
        image_uri: str,
        *,
        job_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RemoteDockerClient:
        """
        Creates a remote environment client from an image.

        Args:
            image_uri: The image uri to create the environment from
            job_id: The job_id of the environment to create
            task_id: The task_id of the environment to create
            metadata: Metadata to associate with the environment

        Returns:
            A tuple containing the remote environment client and the build metadata

        Raises:
            HudResponseError: If the environment creation fails.
        """

        # Validate arguments
        if metadata is None:
            metadata = {}

        logger.info("Creating remote environment")

        true_gym_id = await get_gym_id("docker")

        # augment metadata with dockerfile
        if "environment_config" not in metadata:
            metadata["environment_config"] = {}

        metadata["environment_config"]["image_uri"] = image_uri

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
            raise HudResponseError(
                message=(
                    "Failed to create remote environment: No ID returned in API response. "
                    "Please contact support if this issue persists."
                ),
                response_json=response,
            )

        return cls(env_id)

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


    async def get_mcp_server_endpoint(self) -> str:
        """
        Return the full HTTP URL for the MCP streamable HTTP endpoint of this container.
        """
        raise NotImplementedError("TODO")
