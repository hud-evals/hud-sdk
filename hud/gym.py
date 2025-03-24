from __future__ import annotations
from typing import Any

from hud.environment import Environment

class Gym:
    """
    Represents a simulation environment in the HUD system.

    Attributes:
        id: Unique identifier for the gym
        name: Human-readable name of the gym
    """

    def __init__(self, id: str, name: str, client: Any) -> None:
        """
        Initialize a gym.

        Args:
            id: Unique identifier
            name: Human-readable name
        """
        self.id = id
        self.name = name
        self.client = client

    async def make(self, metadata: dict[str, Any] | None = None) -> Environment:
        """
        Create a new environment for this gym.

        Args:
            metadata: Metadata for the environment

        Returns:
            Environment: The newly created environment
        """

        run = await self.client.create_run(name=self.name, gym=self, metadata=metadata)

        return await run.make()
