"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

import logging

from .telemetry import clear_trace, create_job, get_trace, job, trace

__all__ = [
    "clear_trace",
    "create_job",
    "get_trace",
    "job",
    "trace",
]

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

from .settings import settings

hud_logger = logging.getLogger("hud")
hud_logger.setLevel(logging.INFO)

if settings.hud_logging:
    import sys

    root_logger = logging.getLogger()
    if not root_logger.handlers and not hud_logger.handlers:
        stream = sys.stderr if settings.log_stream.lower() == "stderr" else sys.stdout
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        hud_logger.addHandler(handler)
        hud_logger.propagate = False

try:
    from .utils.agent_patches import apply_all_patches

    apply_all_patches()
except Exception as e:
    logging.getLogger(__name__).debug("Failed to apply agent patches: %s", e)
