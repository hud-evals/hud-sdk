from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class CustomGym(BaseModel):
    """
    Public environment specification with a dockerfile and controller.
    
    If the location is remote, the env will be created on the server.
    If the location is dev, the env will be created locally via docker.
    
    The dockerfile can be specified directly or automatically found in the controller_source_dir.
    If neither is provided, an error will be raised during validation.
    """
    type: Literal["public"] = "public"
    dockerfile: str | None = None
    location: Literal["local", "remote"]
    ports: list[int] | None = None
    # If path, then it is a development environment on the local computer
    # If none, then the controller must be installed in the environment through the dockerfile
    # Can be provided as a string or Path object
    controller_source_dir: str | Path | None = None
    
    def model_post_init(self, __context: Any, /) -> None:
        """Validate and set up dockerfile if not explicitly provided."""
        # Convert string path to Path object if needed
        if isinstance(self.controller_source_dir, str):
            self.controller_source_dir = Path(self.controller_source_dir)
            
        if self.dockerfile is None:
            if self.controller_source_dir is None:
                raise ValueError("Either dockerfile or controller_source_dir must be provided")
            
            # Look for Dockerfile in the controller_source_dir
            dockerfile_path = self.controller_source_dir / "Dockerfile"
            if not dockerfile_path.exists():
                raise ValueError(f"Dockerfile not found in {self.controller_source_dir}")
            
            # Read the Dockerfile content
            self.dockerfile = dockerfile_path.read_text()

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

# Available HUD gyms
ServerGym = Literal["qa", "hud-browser", "hud-ubuntu", "OSWorld-Ubuntu"]

# Gyms can be either custom or server-side
Gym = CustomGym | ServerGym
