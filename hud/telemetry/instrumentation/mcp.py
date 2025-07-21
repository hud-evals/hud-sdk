from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

from mcp.types import (  # type: ignore
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
)
from wrapt import register_post_import_hook, wrap_function_wrapper

from hud.telemetry.context import (
    buffer_mcp_call,
    create_notification_record,
    create_request_record,
    get_current_task_run_id,
)
from hud.telemetry.mcp_models import DirectionType, MCPCallType, MCPManualTestCall, StatusType

logger = logging.getLogger(__name__)


class MCPInstrumentor:
    """
    Context-aware instrumentor for MCP calls.
    Only instruments MCP methods when there's an active trace context.
    """

    def __init__(self) -> None:
        self._installed = False

    def install(self) -> None:
        """Install instrumentation for MCP - but only activate when trace context exists."""
        logger.debug("MCPInstrumentor: install() called (context-aware mode)")
        if self._installed:
            logger.debug("MCP instrumentation already installed")
            return

        # Register hooks but with context-aware wrappers
        hooks = [
            ("mcp.client.sse", "sse_client", self._context_aware_transport_wrapper),
            (
                "mcp.server.sse",
                "SseServerTransport.connect_sse",
                self._context_aware_transport_wrapper,
            ),
            ("mcp.client.stdio", "stdio_client", self._context_aware_transport_wrapper),
            ("mcp.server.stdio", "stdio_server", self._context_aware_transport_wrapper),
            (
                "mcp.shared.session",
                "BaseSession._receive_loop",
                self._context_aware_receive_loop_wrapper,
            ),
        ]

        def create_hook(module: str, func_name: str, wrapper: Any) -> Any:
            return lambda _: wrap_function_wrapper(module, func_name, wrapper)

        for module, func_name, wrapper in hooks:
            logger.debug(
                "MCPInstrumentor: Registering context-aware post-import hook for %s.%s",
                module,
                func_name,
            )
            register_post_import_hook(
                create_hook(module, func_name, wrapper),
                module,
            )

        logger.debug(
            "MCPInstrumentor: Attempting immediate instrumentation of already imported modules "
            "(context-aware)."
        )
        for module, func_name, wrapper in hooks:
            try:
                mod = __import__(module, fromlist=[func_name.split(".")[0]])
                target_obj = mod
                parts = func_name.split(".")
                for i, part in enumerate(parts):
                    if hasattr(target_obj, part):
                        if i == len(parts) - 1:
                            target_obj = getattr(target_obj, part)
                        else:
                            target_obj = getattr(target_obj, part)
                    else:
                        target_obj = None
                        break

                if target_obj and callable(target_obj):
                    logger.debug(
                        "MCPInstrumentor: Wrapping %s.%s (context-aware)", module, func_name
                    )
                    wrap_function_wrapper(module, func_name, wrapper)
                else:
                    logger.debug(
                        "Function %s not found in %s during immediate instrumentation attempt",
                        func_name,
                        module,
                    )
            except ImportError:
                logger.debug("Module %s not imported yet, post-import hook will cover it", module)
            except Exception as e:
                logger.warning("Failed to immediately wrap %s.%s: %s", module, func_name, e)

        try:
            # Import only for testing availability, don't store reference
            __import__("mcp.shared.session")
            wrap_function_wrapper(
                "mcp.shared.session",
                "BaseSession.send_request",
                self._context_aware_send_request_wrapper,
            )
            wrap_function_wrapper(
                "mcp.shared.session",
                "BaseSession.send_notification",
                self._context_aware_send_notification_wrapper,
            )
            logger.debug("Successfully wrapped BaseSession methods for context-aware telemetry")
        except ImportError:
            logger.debug("mcp.shared.session not imported yet, post-import hook will cover it")
        except Exception as e:
            logger.warning("Failed to wrap BaseSession methods: %s", e)

        self._installed = True
        logger.debug("MCP instrumentation installed (context-aware mode)")

    @asynccontextmanager
    async def _context_aware_transport_wrapper(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Context-aware transport wrapper - only logs when trace is active."""
        # Check if we have an active trace context
        hud_task_run_id = get_current_task_run_id()

        if hud_task_run_id:
            original_func_name = f"{wrapped.__module__}.{wrapped.__name__}"
            logger.debug(
                "MCPInstrumentor: _context_aware_transport_wrapper active for %s (trace: %s)",
                original_func_name,
                hud_task_run_id[:8],
            )

        # Always pass through - no interference with MCP handshake
        async with wrapped(*args, **kwargs) as (read_stream, write_stream):
            if hud_task_run_id:
                logger.debug(
                    "Context-aware transport: Yielding instrumented streams for trace %s",
                    hud_task_run_id[:8],
                )
            yield read_stream, write_stream
            if hud_task_run_id:
                logger.debug("Context-aware transport: Exited for trace %s", hud_task_run_id[:8])

    async def _context_aware_receive_loop_wrapper(
        self,
        wrapped: Callable[[Any, Any, Any, Any], Awaitable[Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        """Context-aware receive loop wrapper - only instruments when trace is active."""
        hud_task_run_id = get_current_task_run_id()

        if not hud_task_run_id:
            # No active trace - pass through without instrumentation
            logger.debug("MCPInstrumentor: No active trace - skipping receive_loop instrumentation")
            return await wrapped(*args, **kwargs)

        logger.debug(
            "MCPInstrumentor: Active trace %s - enabling receive_loop instrumentation",
            hud_task_run_id[:8],
        )

        # Only instrument when we have an active trace
        orig_handle_incoming = instance._handle_incoming

        async def handle_incoming_with_telemetry(req_or_msg: Any) -> Any:
            start_time = time.time()
            current_task_run_id = get_current_task_run_id()  # Re-check context

            if not current_task_run_id:
                # Context lost - just pass through
                return await orig_handle_incoming(req_or_msg)

            method_name = "unknown_method"
            message_id = None
            call_type_override = MCPCallType.HANDLE_INCOMING

            actual_message_root = None
            # SessionMessage
            if hasattr(req_or_msg, "message") and hasattr(req_or_msg.message, "root"):
                actual_message_root = req_or_msg.message.root
            elif hasattr(req_or_msg, "root"):  # e.g. RequestResponder
                actual_message_root = req_or_msg.root
            elif isinstance(
                req_or_msg, JSONRPCRequest | JSONRPCNotification | JSONRPCResponse | JSONRPCError
            ):
                actual_message_root = req_or_msg

            if actual_message_root:
                if isinstance(actual_message_root, JSONRPCRequest):
                    method_name = actual_message_root.method
                    message_id = actual_message_root.id
                elif isinstance(actual_message_root, JSONRPCNotification):
                    method_name = actual_message_root.method
                elif isinstance(actual_message_root, JSONRPCResponse | JSONRPCError):
                    message_id = actual_message_root.id

            record_data_base = {
                "method": method_name,
                "message_id": message_id,
                "call_type": call_type_override,
                "direction": DirectionType.RECEIVED,
            }

            try:
                create_request_record(
                    **record_data_base,
                    status=StatusType.STARTED,
                    start_time=start_time,
                )

                result = await orig_handle_incoming(req_or_msg)

                create_request_record(
                    **record_data_base,
                    status=StatusType.COMPLETED,
                    start_time=start_time,
                    end_time=time.time(),
                    duration=time.time() - start_time,
                )
                return result
            except Exception as e:
                create_request_record(
                    **record_data_base,
                    status=StatusType.ERROR,
                    start_time=start_time,
                    end_time=time.time(),
                    duration=time.time() - start_time,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        instance._handle_incoming = handle_incoming_with_telemetry

        # Call original wrapped function with instrumented handler
        return await wrapped(*args, **kwargs)

    async def _context_aware_send_request_wrapper(
        self,
        wrapped: Callable[[Any, Any, Any, Any], Awaitable[Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        """Context-aware send request wrapper."""
        hud_task_run_id = get_current_task_run_id()

        if not hud_task_run_id:
            # No active trace - pass through without instrumentation
            return await wrapped(*args, **kwargs)

        start_time = time.time()
        request = args[0] if args else None
        method_name = request.method if request and hasattr(request, "method") else "unknown_method"
        message_id = request.id if request and hasattr(request, "id") else None

        record_data_base = {
            "method": method_name,
            "message_id": message_id,
            "call_type": MCPCallType.SEND_REQUEST,
            "direction": DirectionType.SENT,
        }

        try:
            create_request_record(
                **record_data_base, status=StatusType.STARTED, start_time=start_time
            )

            result = await wrapped(*args, **kwargs)

            create_request_record(
                **record_data_base,
                status=StatusType.COMPLETED,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
            )
            return result
        except Exception as e:
            create_request_record(
                **record_data_base,
                status=StatusType.ERROR,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def _context_aware_send_notification_wrapper(
        self,
        wrapped: Callable[[Any, Any, Any, Any], Awaitable[Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        """Context-aware send notification wrapper."""
        hud_task_run_id = get_current_task_run_id()

        if not hud_task_run_id:
            # No active trace - pass through without instrumentation
            return await wrapped(*args, **kwargs)

        start_time = time.time()
        notification = args[0] if args else None
        method_name = (
            notification.method
            if notification and hasattr(notification, "method")
            else "unknown_method"
        )

        record_data_base = {
            "method": method_name,
            "message_id": None,  # Notifications don't have IDs
            "call_type": MCPCallType.SEND_NOTIFICATION,
            "direction": DirectionType.SENT,
        }

        try:
            create_notification_record(
                **record_data_base, status=StatusType.STARTED, start_time=start_time
            )

            result = await wrapped(*args, **kwargs)

            create_notification_record(
                **record_data_base,
                status=StatusType.COMPLETED,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
            )
            return result
        except Exception as e:
            create_notification_record(
                **record_data_base,
                status=StatusType.ERROR,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def record_manual_test(self, **kwargs: Any) -> bool:
        """Record a manual test telemetry entry"""
        hud_task_run_id = get_current_task_run_id()
        if not hud_task_run_id:
            logger.warning("Manual test not recorded: No active task_run_id")
            return False
        try:
            record = MCPManualTestCall.create(task_run_id=hud_task_run_id, **kwargs)
            buffer_mcp_call(record)  # buffer_mcp_call handles Pydantic model directly
            logger.debug("Manual test recorded: %s", record.model_dump(exclude_none=True))
            return True
        except Exception as e:
            logger.warning("Manual test not recorded: %s", e)
            return False
