import enum
from pathlib import Path
from typing import Literal, Union
from pydantic import BaseModel


class PrivateEnvSpec(BaseModel):
    """Private environment specification identified by an id."""
    type: Literal["private"] = "private"
    id: str


class PublicEnvSpec(BaseModel):
    """
    Public environment specification with a dockerfile and controller.
    
    If the controller is remote, the env will be created on the server.
    If the controller is dev, the env will be created locally via docker.
    """
    type: Literal["public"] = "public"
    dockerfile: str
    location: Literal["local", "remote"]
    # If path, then it is a development environment on the local computer
    # If str, then it is an environment id (which is a URL to a .tar file)
    controller: Union[Path, str]

# permits either a private or public environment specification
EnvSpec = Union[PrivateEnvSpec, PublicEnvSpec]


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

