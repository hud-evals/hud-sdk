from __future__ import annotations

import asyncio
import concurrent.futures  # For run_coroutine_threadsafe return type
import json
import logging
import threading
import time
from datetime import datetime, timezone  # For ISO timestamp conversion
from typing import TYPE_CHECKING, Any

import httpx

from hud.settings import settings

# Import BaseMCPCall and TrajectoryStep for type hinting and transformation
from hud.telemetry.mcp_models import (  # MCPResponseCall for isinstance check
    BaseMCPCall,
    MCPResponseCall,
    TrajectoryStep,
)

from hud.telemetry.async_logger import AsyncLogger

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from hud.utils.common import Observation

logger = logging.getLogger("hud.telemetry")

# --- Worker Thread and Event Loop Management ---
_worker_thread: threading.Thread | None = None
_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_lock = threading.Lock()  # For protecting worker thread/loop startup
_worker_loop_ready_event = threading.Event()  # Event for sync between threads

# --- Async Queue and Task (managed by the worker loop) ---
_SENTINEL_FOR_WORKER_SHUTDOWN = object()  # Sentinel for queue-based shutdown signaling
_export_queue_async: list[dict[str, Any] | object] = []  # Queue can hold dicts or sentinel
_export_lock_async = asyncio.Lock()  # Async lock for the async queue
_export_task_async: asyncio.Task | None = None  # Async task for processing the queue

# --- Constants ---
EXPORT_INTERVAL = 5.0  # seconds
# MAX_BATCH_SIZE removed as we send one trace payload at a time


def _run_worker_loop() -> None:
    """Target function for the worker thread. Runs its own asyncio event loop."""
    global _worker_loop
    logger.debug("Telemetry worker thread: Starting event loop.")
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)

    _worker_loop_ready_event.set()  # Signal that loop is created and set for this thread

    try:
        logger.debug("Telemetry worker thread: Event loop running.")
        _worker_loop.run_forever()
    except Exception as e:
        logger.exception("Telemetry worker loop encountered an unhandled exception: %s", e)
    finally:
        logger.debug("Telemetry worker loop: Starting cleanup...")
        if _export_task_async and not _export_task_async.done():
            logger.debug("Telemetry worker loop: Cancelling active export processing task.")
            _export_task_async.cancel()
            try:
                # Wait for the task to acknowledge cancellation
                _worker_loop.run_until_complete(
                    asyncio.gather(_export_task_async, return_exceptions=True)
                )
            except asyncio.CancelledError:
                logger.debug(
                    "Telemetry worker loop: Export processing task acknowledged cancellation."
                )
            except Exception as e_gather:
                logger.debug(
                    "Telemetry worker loop: Exception during export task cleanup: %s", e_gather
                )

        logger.debug("Telemetry worker loop: Closing.")
        _worker_loop.close()
        logger.debug("Telemetry worker thread: Event loop closed.")
        # _worker_loop_ready_event.clear() # Should be cleared by starter if thread is to be reused


def _start_worker_if_needed() -> None:
    """Starts the background worker thread if not already running. Assumes _worker_lock is held."""
    global _worker_thread  # _worker_loop is set by the thread itself
    if _worker_thread is None or not _worker_thread.is_alive():
        logger.debug("Telemetry: Worker thread not alive, starting new one.")
        # _worker_loop should be None here or will be replaced by the new thread
        _worker_loop_ready_event.clear()
        _worker_thread = threading.Thread(
            target=_run_worker_loop, daemon=True, name="HUDTelemetryWorker"
        )
        _worker_thread.start()

        logger.debug("Telemetry: Waiting for worker thread event loop to be ready...")
        if not _worker_loop_ready_event.wait(timeout=5.0):  # Wait up to 5 seconds
            logger.error(
                "Telemetry: Worker thread failed to signal event loop readiness within timeout."
            )
            # This is a problem, subsequent submissions might fail.
            return

        # Minor delay to ensure loop might have started run_forever if wait was too tight
        time.sleep(0.05)
        if _worker_loop is None or not _worker_loop.is_running():
            logger.error("Telemetry: Worker loop is not ready or not running after event was set.")
        else:
            logger.debug("Telemetry: Worker thread event loop is ready.")


def submit_to_worker_loop(coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future[Any] | None:
    """Submits a coroutine to be run on the worker thread's event loop."""
    with _worker_lock:  # Protects check-and-start of worker thread/loop
        _start_worker_if_needed()

    # Check _worker_loop status AFTER attempting to start and waiting for readiness event
    if _worker_loop is None or not _worker_loop.is_running():
        logger.error(
            "Telemetry: Worker loop not available or not running for submitting coroutine."
        )
        return None

    try:
        future = asyncio.run_coroutine_threadsafe(coro, _worker_loop)
        return future
    except Exception as e:
        # This can happen if the loop is shut down right as we try to submit
        logger.exception("Telemetry: Failed to submit coroutine to worker loop: %s", e)
        return None


# --- Telemetry Export Logic (runs on worker thread's loop) ---


async def export_telemetry(
    task_run_id: str,
    trace_attributes: dict[str, Any],
    mcp_calls: list[BaseMCPCall],  # Type hint is now list[BaseMCPCall]
) -> None:
    """
    Export telemetry data to the HUD telemetry service.

    Args:
        task_run_id: The task run ID associated with this telemetry
        trace_attributes: Attributes of the trace
        mcp_calls: List of MCP call Pydantic models to export
    """
    trajectory_steps_data: list[dict[str, Any]] = []
    for mcp_call_model in mcp_calls:
        action_data = mcp_call_model.model_dump()

        start_ts_iso = None
        end_ts_iso = None

        # Get start_time if available (e.g. on MCPRequestCall, MCPNotificationCall)
        actual_start_time_float = getattr(mcp_call_model, "start_time", None)
        if actual_start_time_float:
            start_ts_iso = (
                datetime.fromtimestamp(actual_start_time_float, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        # Use 'end_time' if available, otherwise fall back to 'timestamp' for the end_timestamp
        actual_end_time_float = getattr(mcp_call_model, "end_time", None)
        effective_end_timestamp_float = (
            actual_end_time_float if actual_end_time_float else mcp_call_model.timestamp
        )

        if effective_end_timestamp_float:
            end_ts_iso = (
                datetime.fromtimestamp(effective_end_timestamp_float, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        # For events that are more like points in time (e.g., a received response that
        # doesn't have a separate start_time field) set start_timestamp to be the same as
        # end_timestamp if start_timestamp wasn't explicitly set.
        if end_ts_iso and not start_ts_iso:
            start_ts_iso = end_ts_iso

        step_metadata: dict[str, Any] = {
            "mcp_method": mcp_call_model.method,
            "mcp_status": mcp_call_model.status.value,
            "mcp_call_type_original": mcp_call_model.call_type,
        }
        if mcp_call_model.direction:
            step_metadata["mcp_direction"] = mcp_call_model.direction.value
        if mcp_call_model.message_id is not None:
            step_metadata["mcp_message_id"] = str(mcp_call_model.message_id)  # Ensure string

        # Specific handling for MCPResponseCall fields in metadata
        if isinstance(mcp_call_model, MCPResponseCall):
            step_metadata["mcp_is_error"] = mcp_call_model.is_error  # bool is fine for JSON Any
            if mcp_call_model.is_error:
                if mcp_call_model.error is not None:
                    step_metadata["mcp_error_details"] = str(mcp_call_model.error)  # Ensure string
                if mcp_call_model.error_type is not None:
                    step_metadata["mcp_error_type"] = str(
                        mcp_call_model.error_type
                    )  # Ensure string

        obs_text = None
        if isinstance(mcp_call_model, MCPResponseCall) and mcp_call_model.response_data:
            result_data = mcp_call_model.response_data.get("result")
            if result_data is not None:
                try:
                    obs_text = json.dumps(result_data)
                except (TypeError, OverflowError):
                    obs_text = str(result_data)

        trajectory_step = TrajectoryStep(
            type="mcp-step",
            actions=[action_data],
            start_timestamp=start_ts_iso,
            end_timestamp=end_ts_iso,
            metadata=step_metadata,
            observation_text=obs_text,
        )
        trajectory_steps_data.append(trajectory_step.model_dump())

    payload_to_queue = {
        "task_run_id": task_run_id,
        "attributes": trace_attributes,
        "mcp_calls": trajectory_steps_data,
        "timestamp": time.time(),
    }

    await _queue_for_export_async(payload_to_queue)


async def _queue_for_export_async(payload: dict[str, Any] | object) -> None:
    """Adds a payload or sentinel to the async export queue. Runs on worker loop."""
    global _export_task_async, _worker_loop
    if not _worker_loop or not _worker_loop.is_running():
        logger.error("Cannot queue telemetry, worker loop not running or not set.")
        return

    async with _export_lock_async:
        _export_queue_async.append(payload)
        if _export_task_async is None or _export_task_async.done():
            _export_task_async = _worker_loop.create_task(_process_export_queue_async())
            logger.debug("Started/Restarted async telemetry export processing task on worker loop.")


async def _process_export_queue_async() -> None:
    """Processes the async export queue. Runs on worker loop via _export_task_async."""
    global _export_task_async
    try:
        while True:
            payload_to_process: dict[str, Any] | object | None = None
            async with _export_lock_async:
                if not _export_queue_async:
                    logger.debug("Async export queue empty, processing task will pause.")
                    _export_task_async = None
                    return
                payload_to_process = _export_queue_async.pop(0)

            if payload_to_process is _SENTINEL_FOR_WORKER_SHUTDOWN:
                logger.debug("Shutdown sentinel received by processing task, stopping.")
                _export_task_async = None
                return

            if isinstance(payload_to_process, dict):  # Ensure it's a dict before processing as such
                await _export_trace_payload_async(payload_to_process)
            else:
                # Should not happen if only dicts and sentinel are queued
                logger.warning("Unexpected item in telemetry queue: %s", type(payload_to_process))

            await asyncio.sleep(EXPORT_INTERVAL)

    except asyncio.CancelledError:
        logger.debug("Async telemetry export processing task cancelled.")
        _export_task_async = None
        raise
    except Exception as e:
        logger.exception("Error in async telemetry export processing task: %s", e)
        _export_task_async = None


async def _export_trace_payload_async(payload: dict[str, Any]) -> None:
    """Export a single trace payload to the HUD telemetry service."""
    if not settings.telemetry_enabled:
        logger.debug("Telemetry export skipped - telemetry not enabled")
        return

    task_run_id = payload.get("task_run_id")
    if not task_run_id:
        logger.warning("Payload missing task_run_id, skipping export")
        return

    # The payload itself is what we want to send (containing attributes and mcp_calls list)
    # The mcp_calls within the payload are already dumped dictionaries.
    data_to_send = {
        "metadata": payload.get("attributes", {}),
        "telemetry": payload.get("mcp_calls", []),
    }

    await send_telemetry_to_server(task_run_id, data_to_send)


async def send_telemetry_to_server(task_run_id: str, data: dict[str, Any]) -> None:
    telemetry_url = f"{settings.base_url}/v2/task_runs/{task_run_id}/telemetry-upload"

    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.api_key}",
            }

            logger.debug(
                "Exporting telemetry for task run %s to %s",
                task_run_id,
                telemetry_url,
            )
            response = await client.post(
                telemetry_url,
                json=data,  # Send the structured attributes and mcp_calls
                headers=headers,
                timeout=30.0,
            )

            if response.status_code >= 200 and response.status_code < 300:
                logger.debug(
                    "Successfully exported telemetry for task run %s. Status: %s",
                    task_run_id,
                    response.status_code,
                )
            else:
                logger.warning(
                    "Failed to export telemetry for task run %s: HTTP %s - %s",
                    task_run_id,
                    response.status_code,
                    response.text,
                )
    except Exception as e:
        logger.exception("Error exporting telemetry for task run %s: %s", task_run_id, e)


# --- Public Shutdown Function ---
def flush(timeout: float = 10.0) -> None:
    """Flushes pending telemetry data and stops the worker thread."""
    global _worker_thread, _worker_loop, _export_task_async, _export_queue_async
    logger.debug("Initiating telemetry flush and shutdown.")

    shutdown_future: concurrent.futures.Future | None = None
    if _worker_loop and _worker_loop.is_running():
        logger.debug("Submitting shutdown sentinel to telemetry worker's queue.")
        coro = _queue_for_export_async(_SENTINEL_FOR_WORKER_SHUTDOWN)
        try:
            shutdown_future = asyncio.run_coroutine_threadsafe(coro, _worker_loop)
        except Exception as e:  # Catch errors during submission (e.g. if loop is shutting down)
            logger.warning("Exception during submission of shutdown sentinel: %s", e, exc_info=True)
            # Proceed to attempt thread join if possible

        if shutdown_future:
            try:
                shutdown_future.result(timeout / 2 if timeout else None)
                logger.debug("Shutdown sentinel successfully queued.")
            except concurrent.futures.TimeoutError:
                logger.warning("Timeout waiting for shutdown sentinel to be queued.")
            except Exception as e:
                logger.warning(
                    "Error waiting for shutdown sentinel to be queued: %s", e, exc_info=True
                )

    # Wait for the current _export_task_async to see the sentinel and finish.
    # This is tricky because the task lives on another thread's loop.
    # The best way is for _process_export_queue_async to clear _export_task_async when it exits.
    # We then wait a bit for that to happen.
    if _export_task_async is not None:  # Check if a task was even known to be running
        # This check is racy, but it's the best we can do without more complex inter-thread
        # sync for task completion. Give some time for the task to process the sentinel and
        # clear itself.
        # Max wait for task to clear
        attempt_timeout = time.time() + (timeout / 2 if timeout else 2.0)
        while _export_task_async is not None and time.time() < attempt_timeout:
            time.sleep(0.1)
            # _export_task_async is set to None by _process_export_queue_async upon its exit.
        if _export_task_async is not None:
            logger.warning(
                "Telemetry processing task did not clear itself after sentinel. May still be "
                "running or stuck."
            )
        else:
            logger.debug("Telemetry processing task appears to have completed after sentinel.")

    if _worker_loop and _worker_loop.is_running():
        logger.debug("Requesting telemetry worker event loop to stop.")
        # Ask the loop to stop running run_forever
        _worker_loop.call_soon_threadsafe(_worker_loop.stop)

    if _worker_thread and _worker_thread.is_alive():
        logger.debug(
            "Joining telemetry worker thread (up to remaining timeout)...",
        )
        # Calculate remaining timeout for join
        remaining_timeout = timeout - (timeout / 2) if timeout else None  # Simplistic split
        if remaining_timeout is not None and remaining_timeout < 0:
            remaining_timeout = 0

        _worker_thread.join(remaining_timeout)
        if _worker_thread.is_alive():
            logger.warning("Telemetry worker thread did not shut down cleanly after timeout.")
        else:
            logger.debug("Telemetry worker thread successfully joined.")

    _worker_thread = None
    _worker_loop = None
    _export_task_async = None
    # _export_queue_async.clear() # Optionally clear the queue
    logger.debug("Telemetry flush and shutdown process completed.")
