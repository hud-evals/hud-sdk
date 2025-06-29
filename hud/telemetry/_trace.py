from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
    overload,
)

from hud.telemetry import exporter
from hud.telemetry.context import (
    flush_buffer,
    get_current_task_run_id,
    is_root_trace,
    set_current_task_run_id,
)
from hud.telemetry.exporter import submit_to_worker_loop
from hud.telemetry.instrumentation.registry import registry

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Coroutine,
        Generator,
    )

    from hud.telemetry.mcp_models import BaseMCPCall

logger = logging.getLogger("hud.telemetry")
T = TypeVar("T")


def init_telemetry() -> None:
    """Initialize telemetry instrumentors and ensure worker is started if telemetry is active."""
    registry.install_all()
    logger.info("Telemetry initialized.")


@contextmanager
def trace(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """
    Context manager for tracing a block of code.
    The task_run_id is always generated internally as a UUID.
    Telemetry export is handled by a background worker thread.

    Args:
        attributes: Optional dictionary of attributes to associate with this trace
        name: Optional name for this trace, will be added to attributes.

    Returns:
        The generated task run ID (UUID string) used for this trace
    """
    task_run_id = str(uuid.uuid4())

    local_attributes = attributes.copy() if attributes is not None else {}
    if name is not None:
        local_attributes["trace_name"] = name

    start_time = time.time()
    logger.debug("Starting trace %s (Name: %s)", task_run_id, name if name else "Unnamed")

    previous_task_id = get_current_task_run_id()
    was_root = is_root_trace.get()

    set_current_task_run_id(task_run_id)
    is_root = previous_task_id is None
    is_root_trace.set(is_root)

    try:
        yield task_run_id
    finally:
        end_time = time.time()
        duration = end_time - start_time

        mcp_calls: list[BaseMCPCall] = flush_buffer()

        trace_attributes_final = {
            **local_attributes,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "is_root": is_root,
        }

        if is_root and mcp_calls:
            try:
                coro_to_submit = exporter.export_telemetry(
                    task_run_id=task_run_id,
                    trace_attributes=trace_attributes_final,
                    mcp_calls=mcp_calls,
                )
                future = submit_to_worker_loop(coro_to_submit)
                if future:
                    logger.debug(
                        "Telemetry for trace %s submitted to background worker.", task_run_id
                    )
                else:
                    logger.warning(
                        "Failed to submit telemetry for trace %s to"
                        "background worker (loop not available).",
                        task_run_id,
                    )
            except Exception as e:
                logger.warning("Failed to submit telemetry for trace %s: %s", task_run_id, e)

        set_current_task_run_id(previous_task_id)
        is_root_trace.set(was_root)

        logger.debug(
            "Ended trace %s (Name: %s) with %d MCP call(s)",
            task_run_id,
            name if name else "Unnamed",
            len(mcp_calls),
        )

        logger.info("View trace at https://app.hud.so/jobs/traces/%s", task_run_id)


P = ParamSpec("P")
R = TypeVar("R")


def register_trace(
    name: str | None = None, attributes: dict[str, Any] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to wrap a synchronous or asynchronous function call
    within a hud._telemetry.trace context.

    Args:
        name: Optional name for the trace.
        attributes: Optional dictionary of attributes for the trace.
    """

    @overload
    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]: ...

    @overload
    def decorator(func: Callable[P, R]) -> Callable[P, R]: ...

    def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                effective_name = name if name else func.__name__
                with trace(name=effective_name, attributes=attributes):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                effective_name = name if name else func.__name__
                with trace(name=effective_name, attributes=attributes):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
