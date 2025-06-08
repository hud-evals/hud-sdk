from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import textwrap
import time
import uuid
from typing import TYPE_CHECKING, Any

import aiodocker
from aiohttp import ClientTimeout

from hud.env.docker_client import DockerClient, EnvironmentStatus
from hud.server import make_request
from hud.settings import settings
from hud.utils import ExecuteResult
from hud.utils.common import directory_to_tar_bytes, get_gym_id

if TYPE_CHECKING:
    from pathlib import Path

    from aiodocker.containers import DockerContainer
    from aiodocker.stream import Stream

logger = logging.getLogger(__name__)


class LocalDockerClient(DockerClient):
    """
    Docker-based environment client implementation.
    """

    @classmethod
    async def build_image(cls, build_context: Path, verbose: bool = False) -> tuple[str, dict[str, Any]]:
        """
        Build an image from a build context.
        """
        if verbose:
            logging.getLogger("hud").setLevel(logging.DEBUG)
        else:
            logging.getLogger("hud").setLevel(logging.CRITICAL)

        logger.info("Building image from %s", build_context)
        # Create a unique image tag
        image_tag = f"hud-env-{uuid.uuid4().hex[:8]}"

        # Initialize Docker client
        docker_client = aiodocker.Docker()

        # Create a tar file from the path
        tar_bytes = directory_to_tar_bytes(build_context)
        logger.info("generated tar file with size: %d KB", len(tar_bytes) // 1024)

        # Build the image
        build_stream = await docker_client.images.build(
            fileobj=io.BytesIO(tar_bytes),
            encoding="gzip",
            tag=image_tag,
            rm=True,
            pull=True,
            forcerm=True,
        )

        # Print build output
        output = ""
        for chunk in build_stream:
            if "stream" in chunk:
                logger.info(chunk["stream"])
                output += chunk["stream"]

        return image_tag, {"build_output": output}

    @classmethod
    async def create(
        cls,
        image: str,
        host_config: dict[str, Any] | None = None,
        remote_logging_for_local_docker: bool = False,
        *,
        job_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LocalDockerClient:
        """
        Creates a Docker environment client from a image.

        Args:
            image: The image to build the Docker image
            host_config: Optional host configuration for the container
            job_id: Optional job identifier
            task_id: Optional task identifier
            metadata: Optional metadata dictionary

        Returns:
            DockerClient: An instance of the Docker environment client
        """

        # Initialize Docker client
        docker_client = aiodocker.Docker()

        # Default host config
        if host_config is None:
            host_config = {
                "PublishAllPorts": True,
            }

        if remote_logging_for_local_docker:
            # First create remote environment record
            if metadata is None:
                metadata = {}

            # Gym ID for logging local docker is always "local-docker"
            gym_id = await get_gym_id("local-docker")

            if "environment_config" not in metadata:
                metadata["environment_config"] = {}
            metadata["environment_config"]["image_uri"] = image

            # Create a new environment via the HUD API
            response = await make_request(
                method="POST",
                url=f"{settings.base_url}/v2/create_environment",
                json={
                    # still named run_id for backwards compatibility
                    "run_id": job_id,
                    "metadata": metadata,
                    "gym_id": gym_id,
                    "task_id": task_id,
                },
                api_key=settings.api_key,
            )

            # Get the environment ID from the response
            env_id = response.get("id")
            if not env_id:
                raise RuntimeError(f"Failed to create environment on remote: {response}")
        else:
            env_id = None

        # Create and start the container
        container_config = {
            "Image": image,
            "Tty": True,
            "OpenStdin": True,
            "Cmd": None,
            "HostConfig": host_config,
        }

        container = await docker_client.containers.create(config=container_config)

        await container.start()

        # --------------------------------------------------
        # Stream container logs while we wait for readiness
        # --------------------------------------------------
        async def _stream_logs() -> None:
            try:
                # .log() with follow=True -> async iterator of bytes/str
                async for raw in container.log(stdout=True, stderr=True, follow=True):
                    if isinstance(raw, bytes):
                        raw = raw.decode(errors="replace")
                    logger.info("container %s | %s", container.id[:12], raw.rstrip())
            except asyncio.CancelledError:
                # task cancelled during cleanup - silently exit
                logger.info("Log streaming cancelled for container %s", container.id[:12])
                return
            except Exception as e:
                logger.error("error while streaming logs from %s: %s", container.id[:12], str(e))

        log_task: asyncio.Task | None = asyncio.create_task(_stream_logs())

        inspection = await container.show()
        if health_check_config := inspection["Config"].get("Healthcheck"):
            # Using the interval as spinup deadline is a bit implicit - could
            # consider adding explicitly to API if there's demand
            window_usecs = health_check_config.get("Interval", int(30 * 1e9))
            window_secs = window_usecs // 1_000_000

            deadline = time.monotonic() + window_secs
            while True:
                state = (await container.show())["State"]
                health_status = state.get("Health", {}).get("Status")
                container_status = state.get("Status")
                logger.info(
                    "Container %s health status: %s, container status: %s",
                    container.id[:12],
                    health_status,
                    container_status,
                )

                if health_status == "healthy":
                    logger.info("Container %s is healthy", container.id[:12])
                    break
                if container_status in {"exited", "dead"}:
                    logger.error("Container %s crashed before becoming healthy", container.id[:12])
                    raise RuntimeError("Container crashed before becoming healthy")
                now = time.monotonic()
                if now > deadline:
                    logger.error(
                        "Container %s health check timed out after %ds",
                        container.id[:12],
                        window_secs,
                    )
                    raise TimeoutError(f"{container.id} not healthy after {window_secs}s")
                await asyncio.sleep(1)
        else:
            logger.info("Container %s has no healthcheck, assuming ready", container.id[:12])

        # Stop the log stream now that the container is ready
        if log_task is not None:
            logger.info("Cancelling log streaming task for container %s", container.id[:12])
            log_task.cancel()
            with contextlib.suppress(Exception):
                await log_task
            log_task = None

        # Return the controller instance
        logger.info(
            "Creating LocalDockerClient instance for container %s",
            container.id[:12],
        )
        client = cls(docker_client, container.id, env_id)
        # store the task so close() can cancel if it is still running
        client._log_task = log_task  # type: ignore[attr-defined]
        logger.info("LocalDockerClient instance created successfully")
        return client

    def __init__(
        self, docker_conn: aiodocker.Docker, container_id: str, env_id: str | None = None
    ) -> None:
        """
        Initialize the DockerClient.

        Args:
            docker_conn: Docker client connection
            container_id: ID of the Docker container to control
            env_id: ID of the remote environment record
        """
        super().__init__()

        # Store container ID instead of container object
        self._container_id = container_id
        self._env_id = env_id

        # Docker client will be initialized when needed
        self._docker = docker_conn

        # Background task for streaming logs (may be None)
        self._log_task: asyncio.Task | None = None

    @property
    def container_id(self) -> str:
        """Get the container ID."""
        return self._container_id

    @container_id.setter
    def container_id(self, value: str) -> None:
        """Set the container ID."""
        self._container_id = value

    @property
    def env_id(self) -> str | None:
        """Get the environment ID."""
        return self._env_id

    async def _get_container(self) -> DockerContainer:
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

    async def execute(
        self,
        command: list[str],
        *,
        timeout: int | None = None,
    ) -> ExecuteResult:
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
        )
        output: Stream = exec_result.start(timeout=ClientTimeout(timeout), detach=False)

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

        if "No module named 'hud_controller'" in stderr_data.decode():
            if self._source_path is None:
                message = textwrap.dedent("""\
                Your environment is not set up correctly.
                You are using a prebuilt image, so please ensure the following:
                1. Your image cannot be a generic python image, it must contain a python package
                   called hud_controller.
                """)
            else:
                message = textwrap.dedent("""\
                Your environment is not set up correctly.
                You are using a local controller, so please ensure the following:
                1. Your package name is hud_controller
                2. You installed the package in the Dockerfile.
                3. The package is visible from the global python environment (no venv, conda, or uv)
                """)
            logger.error(message)

        return ExecuteResult(
            stdout=bytes(stdout_data),
            stderr=bytes(stderr_data),
            # TODO: Get the exit code from the output
            exit_code=0,
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
        if not isinstance(fileobj, io.BytesIO):
            raise TypeError("fileobj is not a BytesIO object")
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
        Also closes the remote environment record.
        """
        try:
            container = await self._get_container()
            await container.stop()
            await container.delete()
        except Exception as e:
            # Log the error but don't raise it since this is cleanup
            logger.warning("Error during Docker container cleanup: %s", e)
        finally:
            await self._docker.close()

        # Cancel background log forwarding first (if still active)
        if self._log_task is not None:
            self._log_task.cancel()
            with contextlib.suppress(Exception):
                await self._log_task

        # Close the remote environment record
        try:
            if self.env_id is not None:
                await make_request(
                    method="POST",
                    url=f"{settings.base_url}/v2/environments/{self.env_id}/close",
                    api_key=settings.api_key,
                )
        except Exception as e:
            # Log the error but don't raise it since this is cleanup
            logger.warning("Error during remote environment record cleanup: %s", e)
