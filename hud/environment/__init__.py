"""Environment module for HUD SDK.

This module provides different environment implementations for interacting
with either local Docker containers or remote HUD servers.
"""

from hud.environment.base import (
    Environment,
    EnvironmentStatus,
    Observation,
    TaskResult,
    status_messages,
)
from hud.environment.local import LocalEnvironment
from hud.environment.remote import RemoteEnvironment

__all__ = [
    "Environment",
    "EnvironmentStatus",
    "LocalEnvironment",
    "Observation",
    "RemoteEnvironment",
    "TaskResult",
    "status_messages",
]
