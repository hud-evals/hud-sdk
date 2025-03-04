"""
HUD Gym SDK - A Python SDK for interacting with HUD environments.
"""

from __future__ import annotations

from hud.client import HUDClient
from hud.env import Env, EvalSet, Observation, TaskResult
from hud.gym import Gym
from hud.run import Run

__version__ = "0.1.0"

__all__ = [
    "Env",
    "EvalSet",
    "Gym",
    "HUDClient",
    "Observation",
    "Run",
    "TaskResult",
]
