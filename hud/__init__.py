"""
HUD SDK for interacting with the HUD evaluation platform.
"""

from __future__ import annotations

import logging

from . import agent, env, gym, settings, task, taskset, types, utils
from .adapters import ResponseAction as Response
from .job import create_job, load_job, run_job
from .job import job as register_job
from .task import Task
from .taskset import load_taskset
from .telemetry import flush, init_telemetry, trace
from .version import __version__

init_telemetry()

hud_logger = logging.getLogger("hud")
hud_logger.setLevel(logging.INFO)

if not hud_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    hud_logger.addHandler(handler)

__all__ = [
    "Response",
    "Task",
    "__version__",
    "agent",
    "create_job",
    "env",
    "flush",
    "gym",
    "load_job",
    "load_taskset",
    "register_job",
    "run_job",
    "settings",
    "task",
    "taskset",
    "trace",
    "types",
    "utils",
]
