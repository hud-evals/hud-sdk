from __future__ import annotations

import asyncio
import io
import logging
import textwrap
import time
import uuid
from typing import TYPE_CHECKING, Any

import aiodocker
from aiohttp import ClientTimeout

from hud.env.docker_client import DockerClient, EnvironmentStatus
from hud.utils import ExecuteResult
from hud.utils.common import directory_to_tar_bytes

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
    async def build_image(cls, build_context: Path) -> tuple[str, dict[str, Any]]:
        """
        Build an image from a build context.
        """
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
    ) -> LocalDockerClient:
        """
        Creates a Docker environment client from a image.

        Args:
            image: The image to build the Docker image

        Returns:
            DockerClient: An instance of the Docker environment client
        """

        # Initialize Docker client
        docker_client = aiodocker.Docker()

        # Create and start the container
        container_config = {
            "Image": image,
            "Tty": True,
            "OpenStdin": True,
            "Cmd": None,
            "HostConfig": {
                "PublishAllPorts": True,
            },
        }

        container = await docker_client.containers.create(config=container_config)
        await container.start()

        inspection = await container.show()
        if health_check_config := inspection["Config"].get("Healthcheck"):
            # Using the interval as spinup deadline is a bit implicit - could
            # consider adding explicitly to API if there's demand
            window_usecs = health_check_config.get("Interval", int(30 * 1e9))
            window_secs = window_usecs // 1_000_000

            deadline = time.monotonic() + window_secs
            logger.debug("Waiting for container %s to become healthy", container.id)
            while True:
                state = (await container.show())["State"]
                if state.get("Health", {}).get("Status") == "healthy":
                    break
                if state.get("Status") in {"exited", "dead"}:
                    raise RuntimeError("Container crashed before becoming healthy")
                now = time.monotonic()
                if now > deadline:
                    raise TimeoutError(f"{container.id} not healthy after {window_secs}s")
                await asyncio.sleep(1)
            logger.debug("Container %s is healthy", container.id)

        # Return the controller instance
        return cls(docker_client, container.id)

    def __init__(self, docker_conn: aiodocker.Docker, container_id: str) -> None:
        """
        Initialize the DockerClient.

        Args:
            docker_conn: Docker client connection
            container_id: ID of the Docker container to control
        """
        super().__init__()

        # Store container ID instead of container object
        self._container_id = container_id

        # Docker client will be initialized when needed
        self._docker = docker_conn

    @property
    def container_id(self) -> str:
        """Get the container ID."""
        return self._container_id

    @container_id.setter
    def container_id(self, value: str) -> None:
        """Set the container ID."""
        self._container_id = value

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
