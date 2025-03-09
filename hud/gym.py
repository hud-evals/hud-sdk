from __future__ import annotations


class Gym:
    """
    Represents a simulation environment in the HUD system.

    Attributes:
        id: Unique identifier for the gym
        name: Human-readable name of the gym
    """

    def __init__(self, id: str, name: str) -> None:
        """
        Initialize a gym.

        Args:
            id: Unique identifier
            name: Human-readable name
        """
        self.id = id
        self.name = name
