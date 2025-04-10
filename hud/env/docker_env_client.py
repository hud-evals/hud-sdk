from __future__ import annotations

import io
import json
import logging
import tarfile
import tempfile
import uuid
from io import BytesIO
from typing import IO, TYPE_CHECKING, Any

import aiodocker
from aiohttp import ClientTimeout

from hud.env.env_client import EnvClient, EnvironmentStatus
from hud.utils import ExecuteResult

if TYPE_CHECKING:
    from aiodocker.containers import DockerContainer
    from aiodocker.stream import Stream

    from hud.utils.config import ExpandedConfig

logger = logging.getLogger("hud.env.docker_env_client")


def invoke_template(config: ExpandedConfig, package_name: str, divider: str) -> str:
    """
    Return a python script to run the given config.
    """
    func_parts = config.function.split(".")
    module_str = ".".join([package_name] + func_parts[:-1])
    func_str = func_parts[-1]

    # the reason we call `json.dumps` twice is to escape the json string
    return f"""import json
from {module_str} import {func_str}
args = json.loads({json.dumps(json.dumps(config.args))})
result = {func_str}(*args)
result_str = json.dumps(result)
print("{divider}")
print(result_str)
"""


class InvokeError(Exception):
    """
    Error raised when an invoke fails.
    """


def mktar_from_dockerfile(fileobj: BytesIO | IO[bytes]) -> IO[bytes]:
    """
    Create a zipped tar archive from a Dockerfile
    **Remember to close the file object**
    Args:
        fileobj: a Dockerfile
    Returns:
        a NamedTemporaryFile() object
    """
    with tempfile.NamedTemporaryFile() as f, tarfile.open(mode="w:gz", fileobj=f) as t:
        if isinstance(fileobj, io.BytesIO):
            dfinfo = tarfile.TarInfo("Dockerfile")
            dfinfo.size = len(fileobj.getvalue())
            fileobj.seek(0)
        else:
            dfinfo = t.gettarinfo(fileobj=fileobj, arcname="Dockerfile")

        t.addfile(dfinfo, fileobj)
    f.seek(0)
    return f


class DockerEnvClient(EnvClient):
    """
    Docker-based environment client implementation.
    """

    @classmethod
    async def create(cls, dockerfile: str) -> DockerEnvClient:
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

        # Create a tar file from the dockerfile
        dockerfile_tar = mktar_from_dockerfile(dockerfile_fileobj)

        # Build the image
        build_stream = await docker_client.images.build(
            fileobj=dockerfile_tar,
            encoding="gzip",
            tag=image_tag,
            rm=True,
            pull=True,
            forcerm=True,
        )

        # Print build output
        for chunk in build_stream:
            if "stream" in chunk:
                pass

        # Create and start the container
        container_config = {
            "Image": image_tag,
            "Tty": True,
            "OpenStdin": True,
            "Cmd": ["/bin/bash"],
            "HostConfig": {
                "AutoRemove": True,
            },
        }

        container = await docker_client.containers.create(config=container_config)
        await container.start()

        # Return the controller instance
        return cls(docker_client, container.id)

    def __init__(self, docker_conn: aiodocker.Docker, container_id: str) -> None:
        """
        Initialize the DockerEnvClient.

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
        workdir: str | None = None,
        timeout: float | None = None,
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
            workdir=workdir,
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

    async def invoke(self, config: ExpandedConfig) -> tuple[Any, bytes, bytes]:
        """
        Invoke a function in the container.
        """

        if await self.needs_update():
            logger.info("Environment needs update, updating")
            await self.update()

        # generate a random uuid as a divider
        divider = str(uuid.uuid4())

        template = invoke_template(config, self.package_name, divider)
        logger.debug("Invoking template: %s", template)

        result = await self.execute(["python", "-c", template])

        # parse the result
        # we take the whole stderr as the stderr, and the stdout is the result pre-divider
        stderr = result["stderr"]
        stdout_parts = result["stdout"].split(divider.encode())
        stdout = stdout_parts[0]

        # parse the json part of the stdout (if it exists)
        if len(stdout_parts) > 1:
            result = json.loads(stdout_parts[1])
        else:
            raise InvokeError(stdout, stderr)

        return result, stdout, stderr

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
