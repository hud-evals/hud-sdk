from __future__ import annotations

import abc
import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import toml

from hud.env.client import Client
from hud.types import EnvironmentStatus
from hud.utils.common import directory_to_tar_bytes

if TYPE_CHECKING:
    from hud.utils import ExecuteResult
    from hud.utils.config import FunctionConfig

logger = logging.getLogger("hud.env.docker_client")

STATUS_MESSAGES = {
    EnvironmentStatus.RUNNING.value: "is running",
    EnvironmentStatus.ERROR.value: "had an error initializing",
    EnvironmentStatus.COMPLETED.value: "completed",
}

PACKAGE_NAME = "hud_controller"
HUD_MCP_PORT = 8483


class InvokeError(Exception):
    """
    Error raised when an invoke fails.
    """


def invoke_template(config: FunctionConfig, package_name: str, divider: str) -> str:
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


class DockerClient(Client):
    """
    Base class for environment clients.

    Handles updating the environment when local files change.
    """

    _last_requirements_str: str | None = None
    _last_update_time: int = 0
    _last_file_mtimes: dict[str, float] = {}  # noqa: RUF012 - Not recognized as Pydantic model
    _source_path: Path | None = None

    @property
    def source_path(self) -> Path | None:
        """Get the source path."""
        return self._source_path

    def set_source_path(self, source_path: Path) -> None:
        """
        Set the source path for this environment controller.
        Can only be set once, and cannot be set if source_path is already set.

        Args:
            source_path: Path to the source code to use in the environment

        Raises:
            ValueError: If source_path has already been set
        """
        if self._source_path:
            raise ValueError("Source path has already been set")

        # Validate source path
        if not source_path.exists():
            raise FileNotFoundError(f"Source path {source_path} does not exist")
        if not source_path.is_dir():
            raise NotADirectoryError(f"Source path {source_path} is not a directory")
        self._source_path = source_path

        # set current mtimes
        self._last_file_mtimes = self._get_all_file_mtimes()

    @classmethod
    @abc.abstractmethod
    async def build_image(cls, build_context: Path) -> tuple[str, dict[str, Any]]:
        """
        Build an image from a build context.

        Returns:
            tuple[str, dict[str, Any]]: The image tag and build output
        """

    @classmethod
    @abc.abstractmethod
    async def create(cls, image: str) -> DockerClient:
        """
        Creates an environment client from an image.

        Args:
            image: The image to build the environment from

        Returns:
            EnvClient: An instance of the environment client
        """

    @abc.abstractmethod
    async def get_status(self) -> EnvironmentStatus:
        """
        Get the current status of the environment.

        Returns:
            EnvironmentStatus: A status enum indicating the current state of the environment
        """

    def _get_all_file_mtimes(self) -> dict[str, float]:
        """
        Get modification times for all files in the source path.

        Returns:
            Dict[str, float]: Dictionary mapping file paths to modification times
        """
        if not self._source_path:
            return {}

        file_mtimes = {}
        for root, _, files in os.walk(self._source_path):
            for file in files:
                file_path = Path(root) / file
                try:
                    file_mtimes[str(file_path)] = file_path.stat().st_mtime
                except (FileNotFoundError, PermissionError):
                    # Skip files that can't be accessed
                    continue
        return file_mtimes

    async def needs_update(self) -> bool:
        """
        Check if the environment needs an update by:
        1. Checking if any file has been modified since the last update

        Returns:
            bool: True if the environment needs an update, False otherwise.
        """
        # If no source path, no update needed
        if not self.source_path:
            return False

        # Check if any file has been modified since the last update
        current_mtimes = self._get_all_file_mtimes()

        # If we don't have previous modification times, we need an update
        if not self._last_file_mtimes:
            return True

        # Check for new or modified files
        for file_path, mtime in current_mtimes.items():
            if file_path not in self._last_file_mtimes or mtime > self._last_file_mtimes[file_path]:
                return True

        return False

    async def update(self) -> None:
        """
        Base update method for environment controllers.
        For self-managed controllers and controllers with no source path, this is a no-op.
        """
        # TODO: self-managed controllers 

        # If no source path, nothing to update
        if not self._source_path:
            return

        logger.info("Updating environment")

        # Save current file modification times
        self._last_file_mtimes = self._get_all_file_mtimes()

        # Create tar archive of the source code and send it to the container
        tar_bytes = directory_to_tar_bytes(self._source_path / "controller")
        await self.execute(["mkdir", "-p", "/hud/controller"], timeout=5)
        await self.put_archive("/hud/controller", tar_bytes)

        # Check if requirements.txt exists and parse it
        requirements_path = self._source_path / "requirements.txt"
        if not requirements_path.exists():
            raise FileNotFoundError(f"requirements.txt not found in {self._source_path}")

        # Read and parse the current content of pyproject.toml
        current_requirements_content = requirements_path.read_text()
        if (
            self._last_requirements_str is None
            or self._last_requirements_str != current_requirements_content
        ):
            logger.info("Installing requirements from %s", requirements_path.absolute())
            result = await self.execute(
                ["bash", "-c", "cd /hud && uv pip install --requirements requirements.txt"],
                timeout=60,
            )
            if result["stdout"]:
                logger.debug("STDOUT:\n%s", result["stdout"])
            if result["stderr"]:
                logger.debug("STDERR:\n%s", result["stderr"])
            self._last_requirements_str = current_requirements_content

    @abc.abstractmethod
    async def execute(
        self,
        command: list[str],
        *,
        timeout: int | None = None,
    ) -> ExecuteResult:
        """
        Execute a command in the environment. May not be supported by all environments.

        Args:
            command: The command to execute
            timeout: The timeout for the command

        Returns:
            ExecuteResult: The result of the command
        """

    async def invoke(self, config: FunctionConfig) -> tuple[Any, bytes, bytes]:
        """
        Invoke a function in the environment. Supported by all environments.

        Args:
            config: The configuration to invoke

        Returns:
            tuple[Any, bytes, bytes]: The result of the invocation, stdout, and stderr
        """
        if await self.needs_update():
            logger.info("Environment needs update, updating")
            await self.update()

        async with streamablehttp_client("http://localhost:32858/mcp") as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tool_result = await session.call_tool("step", {"action": config})

        return tool_result, b"", b""

        # # generate a random uuid as a divider
        # divider = str(uuid.uuid4())

        # template = invoke_template(config, PACKAGE_NAME, divider)
        # logger.debug("Invoking template: %s", template)

        # result = await self.execute(["python3", "-c", template])

        # # parse the result
        # # we take the whole stderr as the stderr, and the stdout is the result pre-divider
        # stderr = result["stderr"]
        # stdout_parts = result["stdout"].split(divider.encode())
        # stdout = stdout_parts[0]

        # # parse the json part of the stdout (if it exists)
        # if len(stdout_parts) > 1:
        #     result = json.loads(stdout_parts[1])
        # else:
        #     raise InvokeError(stdout, stderr)

        # return result, stdout, stderr

    @abc.abstractmethod
    async def get_archive(self, path: str) -> bytes:
        """
        Get an archive of a path from the environment.
        May not be supported by all environments. (notably browser environments)
        Args:
            path: The path to get the archive of

        Returns:
            bytes: The archive of the path
        """

    @abc.abstractmethod
    async def put_archive(self, path: str, data: bytes) -> bool:
        """
        Put an archive of data at a path in the environment.
        May not be supported by all environments. (notably browser environments)
        Args:
            path: The path to put the archive at
            data: The data to put in the archive
        """
