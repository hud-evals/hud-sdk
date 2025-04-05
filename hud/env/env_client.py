import enum
from pathlib import Path
import tempfile
import uuid
from aiodocker.stream import Stream
from aiohttp import ClientTimeout
from pydantic import BaseModel
import abc
import os
import time
import asyncio
from typing import Literal, Optional, Union, Dict, Any
from hud.utils import ExecuteResult
from hud.utils.common import directory_to_tar_bytes
import io
import aiodocker
from base64 import b64decode, b64encode
from hud.server import make_request
from hud.settings import settings
import toml

# Environment Controllers
class EnvironmentStatus(str, enum.Enum):
    """
    Status of the environment.

    Attributes:
        INITIALIZING: The environment is initializing
        RUNNING: The environment is running
        COMPLETED: The environment is completed
        ERROR: The environment is in an error state
    """

    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"




class EnvClient(BaseModel, abc.ABC):
    """
    Base class for environment clients.
    
    Handles updating the environment when local files change.
    """
    
    def __init__(self):
        """Initialize the base environment controller."""
        # Last known pyproject.toml content
        self.last_pyproject_toml_str = None
        
        # Track the last update time and file modification times
        self.last_update_time = 0
        self.last_file_mtimes: Dict[str, float] = {}
        
        # Source path and env_id are initially None
        self._source_path = None
        self._env_id = None
        self._is_configured = False
        self._package_name = None
        
    @property
    def source_path(self) -> Optional[Path]:
        """Get the source path."""
        return self._source_path
    
    @property
    def env_id(self) -> Optional[str]:
        """Get the environment ID."""
        return self._env_id
    
    @property
    def package_name(self) -> str:
        """Get the package name."""
        if not self._package_name:
            raise ValueError("Package name not set")
        return self._package_name
    

    def set_source_path(self, source_path: Path) -> None:
        """
        Set the source path for this environment controller.
        Can only be set once, and cannot be set if env_id is already set.
        
        Args:
            source_path: Path to the source code to use in the environment
            
        Raises:
            ValueError: If source_path or env_id has already been set
        """
        if self._is_configured:
            raise ValueError("Source path or environment ID has already been set")
        
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
        self._is_configured = True
    
    def set_env_id(self, env_id: str) -> None:
        """
        Set the environment ID for this environment controller.
        Can only be set once, and cannot be set if source_path is already set.
        
        Args:
            env_id: ID of the environment to control
            
        Raises:
            ValueError: If source_path or env_id has already been set
        """
        if self._is_configured:
            raise ValueError("Source path or environment ID has already been set")
        
        self._env_id = env_id
        self._is_configured = True
    
    @classmethod
    @abc.abstractmethod
    async def create(cls, dockerfile: str) -> 'EnvClient':
        """
        Creates an environment client from a dockerfile.

        Args:
            dockerfile: The dockerfile content to build the environment

        Returns:
            EnvClient: An instance of the environment client
        """
        pass
    
    @abc.abstractmethod
    async def get_status(self) -> EnvironmentStatus:
        """
        Get the current status of the environment.
        
        Returns:
            EnvironmentStatus: A status enum indicating the current state of the environment
        """
        pass
    
    def _get_all_file_mtimes(self) -> Dict[str, float]:
        """
        Get modification times for all files in the source path.
        
        Returns:
            Dict[str, float]: Dictionary mapping file paths to modification times
        """
        if not self.source_path:
            return {}
            
        file_mtimes = {}
        for root, _, files in os.walk(self.source_path):
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
        if not self.last_file_mtimes:
            return True
        
        # Check for new or modified files
        for file_path, mtime in current_mtimes.items():
            if file_path not in self.last_file_mtimes or mtime > self.last_file_mtimes[file_path]:
                return True
                
        return False
    
    async def update(self) -> None:
        """
        Base update method for environment controllers.
        For controllers with no source path, this is a no-op.
        """
        # If no source path, nothing to update
        if not self.source_path:
            return
        
        # Save current file modification times
        self.last_file_mtimes = self._get_all_file_mtimes()
        
        # Create tar archive of the source code and send it to the container
        tar_bytes = directory_to_tar_bytes(self.source_path)
        await self.execute(["mkdir", "-p", "/root/controller"], workdir=None, timeout=5)        
        await self.put_archive("/root/controller", tar_bytes)
        
        # Check if pyproject.toml exists and parse it
        pyproject_path = self.source_path / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found in {self.source_path}")
            
        # Read and parse the current content of pyproject.toml
        current_pyproject_content = pyproject_path.read_text()
        if self.last_pyproject_toml_str is None or self.last_pyproject_toml_str != current_pyproject_content:
            # Update package name if pyproject.toml changed
            pyproject_data = toml.loads(current_pyproject_content)
            self._package_name = pyproject_data.get("project", {}).get("name")
            if not self._package_name:
                raise ValueError("Could not find package name in pyproject.toml")
                
            await self.execute(["pip", "install", "-e", "."], workdir="/root/controller", timeout=60)
            # Save current pyproject.toml content
            self.last_pyproject_toml_str = current_pyproject_content
    
    
    @abc.abstractmethod
    async def execute(self, command: list[str], *, workdir: Optional[str] = None, timeout: Optional[float] = None) -> ExecuteResult:
        """Execute a command in the environment."""
        pass
    
    @abc.abstractmethod
    async def get_archive(self, path: str) -> bytes:
        """Get an archive of a path from the environment."""
        pass
    
    @abc.abstractmethod
    async def put_archive(self, path: str, data: bytes) -> bool:
        """Put an archive of data at a path in the environment."""
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the environment and clean up any resources.
        This method should be called when the environment is no longer needed.
        """
        pass
