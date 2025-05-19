from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import threading
import time
import uuid
import sys
import atexit
import weakref
from typing import TYPE_CHECKING, Any, ClassVar

import aiodocker
from aiohttp import ClientTimeout

from hud.env.docker_client import (
    HUD_CONTROLLER_LOG_PATH,
    HUD_MCP_PORT,
    DockerClient,
    EnvironmentStatus,
)
from hud.utils import ExecuteResult
from hud.utils.common import directory_to_tar_bytes

if TYPE_CHECKING:
    from pathlib import Path

    from aiodocker.containers import DockerContainer
    from aiodocker.stream import Stream


logger = logging.getLogger(__name__)


async def stream_mcp_logs(
    container_id: str, health_event: asyncio.Event, error_message_ref: list[str]
) -> None:
    """Stream logs from the MCP server and monitor for errors.

    Args:
        container_id: Docker container ID
        health_event: Event to set when MCP is healthy
        error_message_ref: List to store error message if MCP fails
    """
    docker = aiodocker.Docker()
    try:
        ctr = await docker.containers.get(container_id)

        exec_inst = await ctr.exec(
            cmd=[
                "/bin/bash",
                "-c",
                f"cd /hud/ && uv run -m controller.server 2>&1 | tee {HUD_CONTROLLER_LOG_PATH}",
            ],
            stdout=True,
            stderr=True,
            tty=False,
        )

        logger.debug("Running exec command")
        stream = exec_inst.start(detach=False)

        stdout_data = bytearray()
        stderr_data = bytearray()

        # Wait for up to 30 seconds for the MCP server to start
        start_time = time.monotonic()
        timeout = 30  # seconds

        try:
            while True:
                if time.monotonic() - start_time > timeout:
                    error_message_ref[0] = f"MCP server failed to start within {timeout} seconds"
                    break

                msg = await stream.read_out()
                logger.debug("Received message: %s", msg)
                if msg is None:  # EOF
                    # If we get EOF too early, it might indicate the process crashed
                    exec_info = await exec_inst.inspect()
                    if exec_info.get("ExitCode", 0) != 0:
                        error_message_ref[0] = (
                            f"MCP server crashed with exit code {exec_info.get('ExitCode')}"
                        )
                    break

                if msg.stream == 1:
                    stdout_data.extend(msg.data)
                    # Check for indication that server is ready
                    if b"Started server process" in msg.data:
                        health_event.set()
                        logger.info("MCP server is running")
                elif msg.stream == 2:
                    stderr_data.extend(msg.data)

                prefix = "MCP stdout: " if msg.stream == 1 else "MCP stderr: "
                logger.debug("%s%s", prefix, msg.data.decode(errors="replace").strip())
        finally:
            # If we exited the loop without setting the health event, something went wrong
            if not health_event.is_set():
                stderr_text = stderr_data.decode(errors="replace")
                stdout_text = stdout_data.decode(errors="replace")

                if not error_message_ref[0]:
                    error_message_ref[0] = (
                        f"MCP server failed to start properly\n\nStdout:\n{stdout_text}\n\nStderr:\n{stderr_text}"
                    )

                # Save full logs to a file if they're large
                if len(stdout_data) + len(stderr_data) > 1024:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".log",
                        delete=False,
                    ) as log_file:
                        formatted_output = f"stdout: {stdout_text}\n\nstderr: {stderr_text}"
                        log_file.write(formatted_output)
                        error_message_ref[0] += f"\n\nFull logs saved to {log_file.name}"

            await stream.close()
    finally:
        await docker.close()


def thread_worker(
    container_id: str, health_event: asyncio.Event, error_message_ref: list[str]
) -> None:
    """Run the log streaming in a separate thread with its own event loop.

    Args:
        container_id: Docker container ID
        health_event: Event to set when MCP is healthy
        error_message_ref: List to store error message if MCP fails
    """
    asyncio.run(stream_mcp_logs(container_id, health_event, error_message_ref))


class LocalDockerClient(DockerClient):
    """
    Docker-based environment client implementation.
    """

    # Track active clients with container_id as key and weakref to client as value
    _active_clients: ClassVar[weakref.WeakValueDictionary] = weakref.WeakValueDictionary()

    @classmethod
    def _cleanup_clients(cls) -> None:
        """
        Clean up any remaining clients at program exit.
        """
        for container_id, client_ref in list(cls._active_clients.items()):
            client = client_ref()
            if client is None:  # Reference has been garbage collected
                continue

            try:
                # Create a new event loop for cleanup
                loop = asyncio.new_event_loop()
                loop.run_until_complete(client.close())
                loop.close()
                logger.info("Successfully cleaned up container %s at exit", container_id)
            except Exception as e:
                logger.warning("Error cleaning up container %s at exit: %s", container_id, e)

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
            "ExposedPorts": {
                f"{HUD_MCP_PORT}/tcp": {},
            },
            "Remove": True,
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

        # Create an event to signal when the MCP server is running
        health_event = asyncio.Event()
        # Use a list to hold error message, if any (to pass by reference)
        error_message = [""]

        # Start the log streaming thread
        log_thread = threading.Thread(
            target=thread_worker, args=(container.id, health_event, error_message), daemon=True
        )
        log_thread.start()

        # Wait for the MCP server to start or fail
        try:
            # Wait for up to 30 seconds for the server to start
            timeout = 30
            start_time = time.monotonic()
            while not health_event.is_set() and time.monotonic() - start_time < timeout:
                await asyncio.sleep(0.1)
                if error_message[0]:
                    raise RuntimeError(error_message[0])

            if not health_event.is_set():
                raise TimeoutError(f"MCP server did not start within {timeout} seconds")
        except Exception as e:
            # If anything goes wrong, make sure to clean up
            await container.stop()
            await container.delete()
            raise e
        finally:
            await docker_client.close()

        # Create the client instance
        client = cls(docker_client, container.id)
        client._log_streaming_thread = log_thread
        client._mcp_health_event = health_event
        client._mcp_error_message = error_message[0] if error_message[0] else None

        # Add to active clients dictionary
        cls._active_clients[container.id] = client

        return client

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

            # Remove from active clients dictionary
            if self.container_id in self.__class__._active_clients:
                del self.__class__._active_clients[self.container_id]

        except Exception as e:
            # Log the error but don't raise it since this is cleanup
            logger.warning("Error during Docker container cleanup: %s", e)
        finally:
            await self._docker.close()

    async def get_mcp_server_endpoint(self) -> str:
        """
        Return the full HTTP URL for the MCP streamable HTTP endpoint of this container.
        """
        container = await self._get_container()
        info = await container.show()
        ports = info.get("NetworkSettings", {}).get("Ports", {})
        binding = ports.get(f"{HUD_MCP_PORT}/tcp")
        if not binding or not isinstance(binding, list):
            raise ValueError(
                f"Container port {HUD_MCP_PORT} is not exposed or bound to a host port"
            )
        host_port = binding[0].get("HostPort")
        if not host_port:
            raise ValueError(f"No host port found for container port {HUD_MCP_PORT}")
        return f"http://localhost:{host_port}/mcp/"


# Register atexit handler to clean up any remaining Docker containers
atexit.register(LocalDockerClient._cleanup_clients)
