from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiodocker

from hud.env.docker_client import (
    HUD_MCP_PORT,
    DockerClient,
    EnvironmentStatus,
)
from hud.utils.common import directory_to_tar_bytes
from hud.utils.docker import execute_command_in_container

if TYPE_CHECKING:
    from aiodocker.containers import DockerContainer

    from hud.utils import ExecuteResult


logger = logging.getLogger(__name__)


class ControllerError(RuntimeError):
    """Exception raised when the MCP server fails to start or crashes."""


class ControllerManager:
    """Manages the lifecycle of an MCP server including startup, monitoring, and shutdown."""

    def __init__(self, container_id: str) -> None:
        """
        Initialize the MCP server manager.

        Args:
            container_id: ID of the Docker container running the MCP server
        """
        self.container_id = container_id
        self._docker = aiodocker.Docker()
        self.ready_event = asyncio.Event()
        self.stop_event = asyncio.Event()

        # Error handling
        self._error_lock = threading.Lock()
        self._error: Exception | None = None

        self._log_thread: threading.Thread | None = None
        self._is_running = False
        self._exec_id: str | None = None
        self._stream: Any = None  # Will be set in start

    @property
    def is_running(self) -> bool:
        """Returns whether the MCP server is running."""
        return self._is_running and self.ready_event.is_set() and self.error is None

    @property
    def error(self) -> Exception | None:
        """Returns the error that occurred during MCP server startup, if any."""
        with self._error_lock:
            return self._error

    def _set_error(self, error: Exception) -> None:
        """Thread-safe method to set an error."""
        with self._error_lock:
            if self._error is None:  # Only set the first error
                self._error = error

    def _clear_error(self) -> None:
        """Thread-safe method to clear the error state."""
        with self._error_lock:
            self._error = None

    async def start(self, timeout_secs: int = 30) -> None:
        """
        Start the MCP server and wait for it to be ready.

        Args:
            timeout_secs: Maximum time to wait for server startup in seconds

        Raises:
            MCPServerError: If the server fails to start
            TimeoutError: If the server doesn't start within the timeout
        """
        # Start the server and stream logs in a background thread
        self._log_thread = threading.Thread(
            target=self._run_server_thread, args=(timeout_secs,), daemon=True
        )
        self._log_thread.start()
        self._is_running = True

        # Wait for the MCP server to start or fail
        start_time = time.monotonic()
        while not self.ready_event.is_set() and time.monotonic() - start_time < timeout_secs:
            await asyncio.sleep(0.1)
            # Check if error was set
            error = self.error
            if error is not None:
                raise error

        if not self.ready_event.is_set():
            error = TimeoutError(f"Controller did not start within {timeout_secs} seconds")
            self._set_error(error)
            raise error

    async def stop(self) -> None:
        """Stop the controller if it's running."""
        # Signal the log thread to stop
        self.stop_event.set()

        container = await self._docker.containers.get(self.container_id)

        result = await execute_command_in_container(
            container,
            command=["/hud/scripts/stop_controller.sh"],
            timeout_secs=5,
        )
        if result["exit_code"] != 0:
            raise RuntimeError(
                f"Failed to stop controller: {result['stderr'].decode(errors='replace')}"
            )
        # import ipdb; ipdb.set_trace()
        self.ready_event.clear()
        self._is_running = False
        self._clear_error()  # Clear any error state
        self._exec_id = None
        self._stream = None

        # Wait for the log thread to terminate
        if self._log_thread and self._log_thread.is_alive():
            self._log_thread.join(timeout=5)
        self._log_thread = None

        # Reset the stop event for potential future use
        self.stop_event.clear()

    def _run_server_thread(self, timeout_secs: int) -> None:
        """
        Run the server in a separate thread with its own event loop.

        This function starts the controller process and streams its logs.
        """
        try:
            asyncio.run(self._start_server_and_stream_logs(timeout_secs))
        except Exception as e:
            # Catch any unhandled exceptions in the thread
            self._set_error(ControllerError(f"Unhandled error in controller thread: {e}"))

    async def _start_server_and_stream_logs(self, timeout_secs: int) -> None:
        """
        Start the controller server and stream its logs.

        This combined function handles both starting the server and monitoring its output.

        Args:
            timeout_secs: Maximum time to wait for server startup in seconds
        """
        docker = aiodocker.Docker()
        try:
            ctr = await docker.containers.get(self.container_id)

            # Start the server process
            exec_inst = await ctr.exec(
                cmd="/hud/scripts/start_controller.sh",
                stdout=True,
                stderr=True,
                tty=False,
            )

            # Store the exec ID for potential later use (e.g., stopping the process)
            exec_info = await exec_inst.inspect()
            self._exec_id = exec_info.get("ID")

            # Start the process and get the stream
            self._stream = exec_inst.start(detach=False)
            logger.debug("Started controller process with exec ID: %s", self._exec_id)

            # Stream logs and monitor for server readiness or errors
            stdout_data = bytearray()
            stderr_data = bytearray()

            start_time = time.monotonic()

            try:
                while True:
                    # Check if stop was requested
                    if self.stop_event.is_set():
                        logger.debug("Stop event detected, terminating log streaming")
                        break

                    if (
                        time.monotonic() - start_time > timeout_secs
                        and not self.ready_event.is_set()
                    ):
                        self._set_error(
                            ControllerError(
                                f"Controller failed to start within {timeout_secs} seconds"
                            )
                        )
                        break

                    # Use a small timeout to periodically check the stop event
                    try:
                        msg = await asyncio.wait_for(self._stream.read_out(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # No data received within timeout, check stop event again
                        continue

                    if msg is None:
                        # If we get EOF too early, it might indicate the process crashed
                        if not self.ready_event.is_set():
                            container = await docker.containers.get(self.container_id)
                            exec_info = await container.exec.inspect(self._exec_id)
                            if exec_info.get("ExitCode", 0) != 0:
                                self._set_error(
                                    ControllerError(
                                        "Controller crashed with exit code"
                                        + exec_info.get("ExitCode")
                                    )
                                )
                        break

                    if msg.stream == 1:
                        stdout_data.extend(msg.data)
                        # Check for indication that server is ready
                        if b"Started server process" in msg.data:
                            self.ready_event.set()
                            logger.info("Controller is running")
                    elif msg.stream == 2:
                        stderr_data.extend(msg.data)

                    prefix = "Server stdout: " if msg.stream == 1 else "Server stderr: "
                    logger.debug("%s%s", prefix, msg.data.decode(errors="replace").strip())
            finally:
                # If we exited the loop without setting the ready event, something went wrong
                # BUT only set an error if we're not in the stopping process
                if (
                    not self.ready_event.is_set()
                    and self.error is None
                    and not self.stop_event.is_set()
                ):
                    stderr_text = stderr_data.decode(errors="replace")
                    stdout_text = stdout_data.decode(errors="replace")

                    error_msg = (
                        "Controller failed to start properly\n\n"
                        + f"Stdout:\n{stdout_text}\n\nStderr:\n{stderr_text}"
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
                            error_msg += f"\n\nFull logs saved to {log_file.name}"

                    self._set_error(ControllerError(error_msg))

                # Close stream if it exists
                if self._stream is not None:
                    await self._stream.close()
        except Exception as e:
            self._set_error(ControllerError(f"Failed to start controller: {e}"))
        finally:
            await docker.close()


class LocalDockerClient(DockerClient):
    """
    Docker-based environment client implementation.
    """

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

        # MCP server manager
        self._controller_manager: ControllerManager | None = None

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

        client = cls(docker_client, container.id)
        await client.start_controller()
        return client

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
            timeout: Timeout for the command execution
        Returns:
            ExecuteResult: Result of the command execution
        """
        container = await self._get_container()
        return await execute_command_in_container(
            container,
            command,
            timeout_secs=timeout,
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

    async def start_controller(self) -> ControllerManager:
        """
        Start the HUD controller in the environment. If a controller is already
        running, it will be stopped and a new one will be started in its place.

        Returns:
            ControllerManager: The controller manager instance
        """
        # Ensure scripts are present and up to date
        container = await self._get_container()
        mkdir_result = await execute_command_in_container(
            container,
            command=["mkdir", "-p", "/hud/scripts"],
            timeout_secs=5,
        )
        if mkdir_result["exit_code"] != 0:
            raise RuntimeError(
                "Failed to create scripts directory: "
                + f"{mkdir_result['stderr'].decode(errors='replace')}"
            )

        tar_bytes = directory_to_tar_bytes(Path(__file__).parent / "scripts")
        await self.put_archive("/hud/scripts", tar_bytes)

        if self._controller_manager is None:
            self._controller_manager = ControllerManager(self.container_id)
            await self._controller_manager.start()
        else:
            await self._controller_manager.stop()
            await self._controller_manager.start()
        return self._controller_manager

    async def close(self) -> None:
        """
        Close the Docker environment by stopping and removing the container.
        """
        try:
            # Stop the controller if it's running
            if self._controller_manager:
                await self._controller_manager.stop()

            container = await self._get_container()
            await container.stop()
            await container.delete()
        except Exception as e:
            # Log the error but don't raise it since this is cleanup
            logger.warning("Error during Docker container cleanup: %s", e)
        finally:
            await self._docker.close()

    async def get_controller_endpoint(self) -> str:
        """
        Return the Streamable-HTTP URL for the controller MCP server
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
