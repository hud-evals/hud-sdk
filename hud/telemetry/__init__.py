"""
HUD telemetry module for capturing and reporting telemetry data from MCP calls.

This module provides functionality to trace MCP calls and export telemetry data
to the HUD platform for analysis.
"""

from __future__ import annotations

from hud.telemetry._trace import init_telemetry, register_trace, trace
from hud.telemetry.context import get_current_task_run_id, reset_context, set_current_task_run_id
from hud.telemetry.exporter import flush

__all__ = [
    "flush",
    "get_current_task_run_id",
    "init_telemetry",
    "register_trace",
    "reset_context",
    "set_current_task_run_id",
    "trace",
]
