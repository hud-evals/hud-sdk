from __future__ import annotations

import contextvars
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, TypeVar

from hud.telemetry.mcp_models import (
    BaseMCPCall,
    MCPManualTestCall,
    MCPNotificationCall,
    MCPRequestCall,
    MCPResponseCall,
    StatusType,
)

logger = logging.getLogger("hud.telemetry")

# Context variables for tracing
current_task_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_run_id", default=None
)
# NEW: Global dictionary for buffering, keyed by task_run_id
_GLOBAL_MCP_CALL_BUFFERS: defaultdict[str, list[BaseMCPCall]] = defaultdict(list)
is_root_trace: contextvars.ContextVar[bool] = contextvars.ContextVar("is_root_trace", default=False)

# Maximum buffer size before automatic flush
MAX_BUFFER_SIZE = 100

# Type variable for record factories
T = TypeVar("T", bound=BaseMCPCall)


def get_current_task_run_id() -> str | None:
    """Get the task_run_id for the current trace context."""
    return current_task_run_id.get()


def set_current_task_run_id(task_run_id: str | None) -> None:
    """Set the task_run_id for the current trace context."""
    current_task_run_id.set(task_run_id)


def buffer_mcp_call(record: BaseMCPCall | dict[str, Any]) -> None:
    task_run_id = get_current_task_run_id()

    if not task_run_id:
        logger.warning(
            "BUFFER_MCP_CALL: No task_run_id. Skipping buffer for %s", type(record).__name__
        )
        return

    # Ensure 'record' is a Pydantic model instance from here
    if isinstance(record, dict):
        try:
            record_model = BaseMCPCall.from_dict(record)
            record = record_model
        except Exception as e_conv:
            logger.exception("BUFFER_MCP_CALL: Failed to convert dict to BaseMCPCall: %s", e_conv)
            return

    _GLOBAL_MCP_CALL_BUFFERS[task_run_id].append(record)
    buffer_len = len(_GLOBAL_MCP_CALL_BUFFERS[task_run_id])

    if buffer_len >= MAX_BUFFER_SIZE:
        flush_buffer(export=True)


def flush_buffer(export: bool = False) -> list[BaseMCPCall]:
    """
    Clear the MCP calls buffer and return its contents.

    Args:
        export: Whether to trigger export of this buffer

    Returns:
        The list of buffered MCP calls
    """
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("FLUSH_BUFFER: No current task_run_id. Cannot flush.")
        return []

    buffer_for_task = _GLOBAL_MCP_CALL_BUFFERS.pop(
        task_run_id, []
    )  # Get and remove the list for this task

    return buffer_for_task  # Return the flushed items


def create_request_record(
    method: str, status: StatusType = StatusType.STARTED, **kwargs: Any
) -> MCPRequestCall:
    """Create and buffer a request record"""
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("No active task_run_id, request record will not be created")
        raise ValueError("No active task_run_id")

    record = MCPRequestCall(
        task_run_id=task_run_id,
        method=method,
        status=status,
        start_time=kwargs.pop("start_time", None) or datetime.now().timestamp(),
        **kwargs,
    )
    buffer_mcp_call(record)
    return record


def create_response_record(
    method: str, related_request_id: str | int | None = None, is_error: bool = False, **kwargs: Any
) -> MCPResponseCall:
    """Create and buffer a response record"""
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("No active task_run_id, response record will not be created")
        raise ValueError("No active task_run_id")

    record = MCPResponseCall(
        task_run_id=task_run_id,
        method=method,
        status=StatusType.COMPLETED,
        related_request_id=related_request_id,
        is_error=is_error,
        **kwargs,
    )

    buffer_mcp_call(record)
    return record


def create_notification_record(
    method: str, status: StatusType = StatusType.STARTED, **kwargs: Any
) -> MCPNotificationCall:
    """Create and buffer a notification record"""
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("No active task_run_id, notification record will not be created")
        raise ValueError("No active task_run_id")

    record = MCPNotificationCall(
        task_run_id=task_run_id,
        method=method,
        status=status,
        start_time=kwargs.pop("start_time", None) or datetime.now().timestamp(),
        **kwargs,
    )
    buffer_mcp_call(record)
    return record


def create_manual_test_record(**custom_data: Any) -> MCPManualTestCall | None:
    """Create and buffer a manual test record"""
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("No active task_run_id, manual test record will not be created")
        return None

    record = MCPManualTestCall.create(task_run_id=task_run_id, **custom_data)
    buffer_mcp_call(record)
    return record


def reset_context() -> None:
    """Reset all telemetry context variables. Useful for test isolation."""
    set_current_task_run_id(None)
    is_root_trace.set(False)
