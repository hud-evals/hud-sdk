from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel


class CustomGym(BaseModel):
    """
    Public environment specification with a dockerfile and controller.

    If the location is remote, the env will be created on the server.
    If the location is local, the env will be created locally via docker.

    The dockerfile can be specified directly or automatically found in the controller_source_dir.
    If neither is provided, an error will be raised during validation.
    """

    type: Literal["public"] = "public"
    location: Literal["local", "remote"]
    # A. If path, then it is a docker build context on the local computer.
    #    If the location is local, docker build will be used to create the image.
    #    If the location is remote, we will build the image remotely.
    #    The controller will be automatically installed and kept in sync with local changes
    #    as long as a pyproject.toml is present at the root of the folder.
    # B. If string, then it is the uri of the docker image to use.
    #    The controller must already be installed in the image.
    image_or_build_context: str | Path
    # host_config will be passed to the docker client when creating the environment.
    # refer to official docker api documentation for available configs.
    host_config: dict[str, Any] | None = None


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
ServerGym: TypeAlias = Literal["qa", "hud-browser", "OSWorld-Ubuntu"]

# Gyms can be either custom or server-side
Gym: TypeAlias = CustomGym | ServerGym

# Metadata keys for the environment.
# partial: Whether the environment evaluator should give partial grades.
# eval_model: The model to use for evaluation when running a VLM. Wraps langchain.
ServerMetadataKeys: TypeAlias = Literal["partial", "eval_model"]
MetadataKeys: TypeAlias = str | ServerMetadataKeys
