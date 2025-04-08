from __future__ import annotations

import abc
import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import toml
from pydantic import BaseModel

from hud.types import EnvironmentStatus
from hud.utils.common import directory_to_tar_bytes

if TYPE_CHECKING:
    from hud.utils import ExecuteResult
    from hud.utils.config import ExpandedConfig

logger = logging.getLogger("hud.env.env_client")

STATUS_MESSAGES = {
    EnvironmentStatus.RUNNING.value: "is running",
    EnvironmentStatus.ERROR.value: "had an error initializing",
    EnvironmentStatus.COMPLETED.value: "completed",
}


class EnvClient(BaseModel):
    """
    Base class for environment clients.
    
    Handles updating the environment when local files change.
    """
    
    _last_pyproject_toml_str: str | None = None
    _last_update_time: int = 0
    _last_file_mtimes: dict[str, float] = {}
    _source_path: Path | None = None
    _package_name: str | None = None

    @property
    def source_path(self) -> Path | None:
        """Get the source path."""
        return self._source_path
    
    @property
    def package_name(self) -> str:
        """Get the package name."""
        if not self._package_name:
            raise ValueError("Package name not set")
        return self._package_name
    

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
        
        # Parse pyproject.toml to get package name
        pyproject_path = source_path / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found in {source_path}")
            
        pyproject_data = toml.load(pyproject_path)
        self._package_name = pyproject_data.get("project", {}).get("name")
        if not self._package_name:
            raise ValueError("Could not find package name in pyproject.toml")
        
        self._source_path = source_path
    
    @classmethod
    @abc.abstractmethod
    async def create(cls, dockerfile: str) -> EnvClient:
        """
        Creates an environment client from a dockerfile.

        Args:
            dockerfile: The dockerfile content to build the environment

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
        For controllers with no source path, this is a no-op.
        """
        # If no source path, nothing to update
        if not self._source_path:
            return
        
        logger.info("Updating environment")

        # Save current file modification times
        self._last_file_mtimes = self._get_all_file_mtimes()
        
        # Create tar archive of the source code and send it to the container
        tar_bytes = directory_to_tar_bytes(self._source_path)
        await self.execute(["mkdir", "-p", "/root/controller"], workdir=None, timeout=5)
        await self.put_archive("/root/controller", tar_bytes)
        
        # Check if pyproject.toml exists and parse it
        pyproject_path = self._source_path / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found in {self._source_path}")
            
        # Read and parse the current content of pyproject.toml
        current_pyproject_content = pyproject_path.read_text()
        if (
            self._last_pyproject_toml_str is None
            or self._last_pyproject_toml_str != current_pyproject_content
        ):
            # Update package name if pyproject.toml changed
            pyproject_data = toml.loads(current_pyproject_content)
            self._package_name = pyproject_data.get("project", {}).get("name")
            if not self._package_name:
                raise ValueError("Could not find package name in pyproject.toml")
            logger.info("Installing %s in /root/controller", self._package_name)
            result = await self.execute(
                ["pip", "install", "-e", "."],
                workdir="/root/controller",
                timeout=60,
            )
            if result["stdout"]:
                logger.info("STDOUT:\n%s", result["stdout"])
            if result["stderr"]:
                logger.warning("STDERR:\n%s", result["stderr"])
            # Save current pyproject.toml content
            self._last_pyproject_toml_str = current_pyproject_content
    
    
    @abc.abstractmethod
    async def execute(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> ExecuteResult:
        """
        Execute a command in the environment. May not be supported by all environments.
        
        Args:
            command: The command to execute
            workdir: The working directory to execute the command in
            timeout: The timeout for the command
            
        Returns:
            ExecuteResult: The result of the command
        """
    
    @abc.abstractmethod
    async def invoke(self, config: ExpandedConfig) -> tuple[Any, bytes, bytes]:
        """
        Invoke a function in the environment. Supported by all environments.
        
        Args:
            config: The configuration to invoke

        Returns:
            tuple[Any, bytes, bytes]: The result of the invocation, stdout, and stderr
        """

    async def wait_for_ready(self) -> None:
        """Wait for the environment to be ready."""
        while True:
            state = await self.get_status()
            if state in (
                EnvironmentStatus.RUNNING.value,
                EnvironmentStatus.ERROR.value,
                EnvironmentStatus.COMPLETED.value,
            ):
                logger.info("Environment %s", STATUS_MESSAGES.get(state))
                break
            await asyncio.sleep(5)


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

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the environment and clean up any resources.
        This method should be called when the environment is no longer needed.
        """
