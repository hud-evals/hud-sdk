from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from hud.types import EnvironmentStatus
    from hud.utils.config import HudStyleConfig


class Client(BaseModel, ABC):
    """
    Base class for all environment clients.
    """

    @abstractmethod
    async def invoke(self, config: HudStyleConfig) -> Any:
        """
        Invoke the environment with the given config.
        """

    @abstractmethod
    async def get_status(self) -> EnvironmentStatus:
        """
        Get the current status of the environment.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Close the environment and clean up any resources.
        This method should be called when the environment is no longer needed.
        """
