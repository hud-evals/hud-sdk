from __future__ import annotations

import contextvars
import logging
from datetime import datetime
from typing import Any, TypeVar

from hud.telemetry.mcp_models import (
    BaseMCPCall,
    MCPManualTestCall,
    MCPNotificationCall,
    MCPRequestCall,
    MCPResponseCall,
    MCPTelemetryRecord,
    StatusType,
)

logger = logging.getLogger("hud.telemetry")

# Context variables for tracing
current_task_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_run_id", default=None
)
mcp_calls_buffer: contextvars.ContextVar[list[BaseMCPCall] | None] = contextvars.ContextVar(
    "mcp_calls_buffer", default=None
)
is_root_trace: contextvars.ContextVar[bool] = contextvars.ContextVar("is_root_trace", default=False)

# Maximum buffer size before automatic flush
MAX_BUFFER_SIZE = 100

# Type variable for record factories
T = TypeVar("T", bound=BaseMCPCall)


def get_current_task_run_id() -> str | None:
    """Get the task_run_id for the current trace context."""
    value = current_task_run_id.get()
    # Convert empty string sentinel back to None
    return None if value == "" else value


def set_current_task_run_id(task_run_id: str | None) -> None:
    """Set the task_run_id for the current trace context."""
    # Handle None value by using empty string as sentinel
    value_to_set = "" if task_run_id is None else task_run_id
    current_task_run_id.set(value_to_set)


def buffer_mcp_call(record: BaseMCPCall | dict[str, Any]) -> None:
    """
    Add an MCP call to the buffer for the current trace.

    Args:
        record: Either a Pydantic model instance or dictionary with MCP call data
    """
    # Only buffer if we have an active trace
    task_run_id = get_current_task_run_id()
    if task_run_id is not None and task_run_id != "":
        buffer = mcp_calls_buffer.get()
        if buffer is None:
            buffer = []

        # Convert dictionary to proper model if needed
        if isinstance(record, dict):
            record = BaseMCPCall.from_dict(record)

        # Ensure the record has the current task_run_id
        if record.task_run_id != task_run_id:
            # Create a copy with the current task_run_id
            record_dict = record.model_dump()
            record_dict["task_run_id"] = task_run_id
            record = BaseMCPCall.from_dict(record_dict)

        # Add to buffer
        buffer.append(record)
        mcp_calls_buffer.set(buffer)

        # Auto-flush if buffer gets too large
        if len(buffer) >= MAX_BUFFER_SIZE:
            logger.debug("MCP calls buffer reached size %d, auto-flushing", len(buffer))
            flush_buffer(export=True)


def flush_buffer(export: bool = False) -> list[BaseMCPCall]:
    """
    Clear the MCP calls buffer and return its contents.

    Args:
        export: Whether to trigger export of this buffer

    Returns:
        The list of buffered MCP calls
    """
    buffer = mcp_calls_buffer.get()
    if buffer is None:
        buffer = []
    # Reset buffer to empty list
    mcp_calls_buffer.set([])

    if export and buffer and len(buffer) > 0:
        task_id = buffer[0].task_run_id if buffer else None
        if task_id:
            logger.debug("Exporting %d MCP calls for task run %s", len(buffer), task_id)
            # Create a telemetry record for export
            _telemetry_record = MCPTelemetryRecord(task_run_id=task_id, records=buffer)
            # In the future, we could call an export function here
            # For now, just log that we have telemetry
            logger.debug("MCP telemetry record created with %d calls", len(buffer))
        else:
            logger.warning("No task_run_id found in buffer, skipping export")

    return buffer


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
    mcp_calls_buffer.set([])
    is_root_trace.set(False)
