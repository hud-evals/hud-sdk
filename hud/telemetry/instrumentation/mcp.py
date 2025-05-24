from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

# from dataclasses import dataclass # Likely unused now
from mcp.shared.message import SessionMessage  # type: ignore

# MCP type imports for type checking and attribute access safety
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
    create_response_record,
    get_current_task_run_id,
)
from hud.telemetry.mcp_models import DirectionType, MCPCallType, MCPManualTestCall, StatusType

logger = logging.getLogger(__name__)

# Ensure no OTel imports remain
# from opentelemetry import context as otel_context, propagate # Should be removed


class MCPInstrumentor:
    """
    Instrumentor for MCP calls.
    """

    def __init__(self) -> None:
        self._installed = False

    def install(self) -> None:
        """Install instrumentation for MCP"""
        logger.debug("MCPInstrumentor: install() called")
        if self._installed:
            logger.debug("MCP instrumentation already installed")
            return

        # Updated hooks: removed _transport_wrapper and _base_session_init_wrapper related hooks
        # if they become no-ops. For now, let's assume _transport_wrapper might still be used,
        # but simplified. _base_session_init_wrapper hook is removed.
        hooks = [
            ("mcp.client.sse", "sse_client", self._transport_wrapper),
            ("mcp.server.sse", "SseServerTransport.connect_sse", self._transport_wrapper),
            ("mcp.client.stdio", "stdio_client", self._transport_wrapper),
            ("mcp.server.stdio", "stdio_server", self._transport_wrapper),
            ("mcp.shared.session", "BaseSession._receive_loop", self._receive_loop_wrapper),
        ]

        def create_hook(module: str, func_name: str, wrapper: Any) -> Any:
            return lambda _: wrap_function_wrapper(module, func_name, wrapper)

        for module, func_name, wrapper in hooks:
            logger.debug(
                "MCPInstrumentor: Registering post-import hook for %s.%s", module, func_name
            )
            register_post_import_hook(
                create_hook(module, func_name, wrapper),
                module,
            )

        # Removed hook for BaseSession.__init__ (it was for _base_session_init_wrapper)

        logger.debug(
            "MCPInstrumentor: Attempting immediate instrumentation of already imported modules."
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
                    logger.debug("MCPInstrumentor: Wrapping %s.%s", module, func_name)
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
                self._send_request_telemetry_wrapper,
            )
            wrap_function_wrapper(
                "mcp.shared.session",
                "BaseSession.send_notification",
                self._send_notification_telemetry_wrapper,
            )
            logger.debug("Successfully wrapped BaseSession methods for telemetry")
        except ImportError:
            logger.debug("mcp.shared.session not imported yet, post-import hook will cover it")
        except Exception as e:
            logger.warning("Failed to wrap BaseSession methods: %s", e)

        self._installed = True
        logger.info("MCP instrumentation installed (simplified)")

    @asynccontextmanager
    async def _transport_wrapper(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Wrap transport functions. Simplified: passes through original streams."""
        original_func_name = f"{wrapped.__module__}.{wrapped.__name__}"
        logger.debug(
            "MCPInstrumentor: _transport_wrapper called for %s (passthrough)", original_func_name
        )

        # No OTel context or HUD Task ID manipulation here anymore for transport layer itself.
        # Higher level wrappers (_send_request, _receive_loop) will handle HUD Task ID.

        async with wrapped(*args, **kwargs) as (read_stream, write_stream):
            logger.debug("_transport_wrapper: Yielding original streams for %s", original_func_name)
            yield read_stream, write_stream  # Pass original streams directly
            logger.debug(
                "_transport_wrapper: Exited original stream context for %s", original_func_name
            )

    async def _receive_loop_wrapper(
        self,
        wrapped: Callable[[Any, Any, Any, Any], Awaitable[Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        """
        Wrap _receive_loop to capture responses and handled incoming messages.
        """
        logger.debug("MCPInstrumentor: _receive_loop_wrapper called")

        orig_handle_incoming = instance._handle_incoming

        async def handle_incoming_with_telemetry(req_or_msg: Any) -> Any:
            start_time = time.time()
            hud_task_run_id = get_current_task_run_id()
            method_name = "unknown_method"
            message_id = None
            is_response_or_error_flag = False
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
                    # call_type_override can remain HANDLE_INCOMING or be more specific if desired
                elif isinstance(actual_message_root, JSONRPCNotification):
                    method_name = actual_message_root.method
                    message_id = None  # Notifications don't have IDs
                    # call_type_override can remain HANDLE_INCOMING
                elif isinstance(actual_message_root, JSONRPCResponse | JSONRPCError):
                    # This case implies _handle_incoming is processing a response/error directly
                    # (e.g. an error encountered while trying to send/route a response previously)
                    message_id = actual_message_root.id
                    method_name = f"internal_response_handling_for_id_{message_id}"
                    is_response_or_error_flag = True
                    # Treat as receiving a response internally
                    call_type_override = MCPCallType.RECEIVE_RESPONSE
                else:
                    # Fallback for other types, if any, that might appear here
                    if hasattr(actual_message_root, "method"):
                        method_name = actual_message_root.method
                    if hasattr(actual_message_root, "id"):
                        message_id = actual_message_root.id

            record_data_base = {
                "method": method_name,
                "direction": DirectionType.RECEIVED,
                "call_type": call_type_override,
                "message_id": message_id,
                "is_response_or_error": is_response_or_error_flag,  # For MCPRequestCall model
            }

            try:
                if hud_task_run_id:
                    create_request_record(
                        **record_data_base,
                        status=StatusType.STARTED,
                        start_time=start_time,
                        # request_data might be populated if needed for HANDLE_INCOMING
                    )

                result = await orig_handle_incoming(req_or_msg)

                if hud_task_run_id:
                    create_request_record(
                        **record_data_base,
                        status=StatusType.COMPLETED,
                        start_time=start_time,
                        end_time=time.time(),
                        duration=time.time() - start_time,
                        # result_data might be populated if needed
                    )
                return result
            except Exception as e:
                if hud_task_run_id:
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

        try:
            logger.debug("MCPInstrumentor: Entering instrumented _receive_loop section")
            # The original logic for distinguishing request/notification from response for
            # telemetry:
            # Ensure streams are context managed
            async with instance._read_stream, instance._write_stream:
                async for message in instance._read_stream:
                    hud_task_run_id = get_current_task_run_id()  # Get ID for each message

                    if isinstance(message, Exception):
                        await instance._handle_incoming(message)  # Will be wrapped
                        continue

                    # Ensure we are dealing with SessionMessage
                    if (
                        not isinstance(message, SessionMessage)
                        or not hasattr(message, "message")
                        or not hasattr(message.message, "root")
                    ):
                        logger.warning(
                            "Unexpected message type in _receive_loop: %s", type(message)
                        )
                        await instance._handle_incoming(
                            RuntimeError(f"Unknown message type: {message}")
                        )
                        continue

                    msg_root = message.message.root

                    if isinstance(msg_root, JSONRPCRequest | JSONRPCNotification):
                        # Let the (wrapped) _handle_incoming deal with these and record telemetry
                        await instance._handle_incoming(message)
                    elif isinstance(msg_root, JSONRPCResponse | JSONRPCError):
                        # This is a direct response/error, record it here
                        logger.debug(
                            "MCPInstrumentor: Capturing direct response/error ID: %s", msg_root.id
                        )
                        if hud_task_run_id:
                            is_error = isinstance(msg_root, JSONRPCError)
                            error_message = None
                            error_code = None
                            if is_error and hasattr(msg_root, "error"):
                                error_message = getattr(msg_root.error, "message", None)
                                error_code = str(getattr(msg_root.error, "code", None))

                            create_response_record(
                                method=f"response_to_id_{msg_root.id}",  # Consistent method naming
                                related_request_id=msg_root.id,
                                response_id=msg_root.id,
                                is_error=is_error,
                                response_data=msg_root.model_dump(exclude_none=True),
                                error=error_message,
                                error_type=error_code,
                                direction=DirectionType.RECEIVED,
                                call_type=MCPCallType.RECEIVE_RESPONSE,
                                timestamp=time.time(),  # Add timestamp here for accuracy
                            )

                        # Original logic to pass response to waiting stream
                        stream = instance._response_streams.pop(msg_root.id, None)
                        if stream:
                            await stream.send(msg_root)
                        else:
                            # This case should ideally be handled by _handle_incoming if it's an
                            # unsolicited response
                            logger.warning(
                                "Received response/error with unknown request ID %s and no "
                                "response stream.",
                                msg_root.id,
                            )
                            # Potentially pass to _handle_incoming if that's desired for unroutable
                            # responses
                            await instance._handle_incoming(message)  # Let wrapped handler decide
                    else:
                        logger.warning(
                            "Unknown message root type in _receive_loop: %s", type(msg_root)
                        )
                        await instance._handle_incoming(message)  # Let wrapped handler decide

            logger.debug("MCPInstrumentor: Exiting instrumented _receive_loop section")
        except Exception as e:
            logger.error("Error in instrumented receive loop: %s, falling back to original", e)
            instance._handle_incoming = orig_handle_incoming
            # Re-raising to ensure original behavior if full loop fails, or call wrapped directly
            # For simplicity now, we just log and let finally restore.
            raise
        finally:
            instance._handle_incoming = orig_handle_incoming
            logger.debug("MCPInstrumentor: Restored original _handle_incoming")

    async def _send_request_telemetry_wrapper(
        self,
        wrapped: Callable[[Any, Any, Any, Any], Awaitable[Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        start_time = time.time()
        hud_task_run_id = get_current_task_run_id()
        method_name = "unknown_method"
        request_obj_data = None
        request_id = None

        if args and len(args) > 0:
            request_session_msg = args[0]  # Assuming SessionMessage
            if hasattr(request_session_msg, "message") and hasattr(
                request_session_msg.message, "root"
            ):
                actual_request = request_session_msg.message.root
                if hasattr(actual_request, "method"):
                    method_name = actual_request.method
                if hasattr(actual_request, "id"):
                    request_id = actual_request.id
                if hasattr(actual_request, "model_dump"):
                    try:
                        request_obj_data = actual_request.model_dump(exclude_none=True)
                    except Exception as e:
                        logger.warning(
                            "Could not dump request data for %s: %s",
                            method_name,
                            e,
                        )

        record_data_base = {
            "method": method_name,
            "direction": DirectionType.SENT,
            "call_type": MCPCallType.SEND_REQUEST,  # More specific type
            "request_id": request_id,
            "message_id": request_id,
            "request_data": request_obj_data,
        }

        try:
            if hud_task_run_id:
                create_request_record(
                    **record_data_base, status=StatusType.STARTED, start_time=start_time
                )

            result = await wrapped(*args, **kwargs)  # result here is usually None for send_request

            if hud_task_run_id:
                # For send_request, the 'result' is the response future, not the response itself.
                # Completion means the request was successfully sent to the transport.
                # The actual response is captured by _receive_loop_wrapper.
                create_request_record(
                    **record_data_base,
                    status=StatusType.COMPLETED,  # Meaning: successfully dispatched
                    start_time=start_time,
                    end_time=time.time(),
                    duration=time.time() - start_time,
                )
            return result
        except Exception as e:
            if hud_task_run_id:
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

    async def _send_notification_telemetry_wrapper(
        self,
        wrapped: Callable[[Any, Any, Any, Any], Awaitable[Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        start_time = time.time()
        hud_task_run_id = get_current_task_run_id()
        method_name = "unknown_method"
        notification_obj_data = None

        if args and len(args) > 0:
            notification_session_msg = args[0]  # Assuming SessionMessage
            if hasattr(notification_session_msg, "message") and hasattr(
                notification_session_msg.message, "root"
            ):
                actual_notification = notification_session_msg.message.root
                if hasattr(actual_notification, "method"):
                    method_name = actual_notification.method
                if hasattr(actual_notification, "model_dump"):
                    try:
                        notification_obj_data = actual_notification.model_dump(exclude_none=True)
                    except Exception as e:
                        logger.warning(
                            "Could not dump notification data for %s: %s",
                            method_name,
                            e,
                        )

        record_data_base = {
            "method": method_name,
            "direction": DirectionType.SENT,
            "call_type": MCPCallType.SEND_NOTIFICATION,  # More specific type
            "notification_data": notification_obj_data,
        }

        try:
            if hud_task_run_id:
                create_notification_record(
                    **record_data_base, status=StatusType.STARTED, start_time=start_time
                )

            result = await wrapped(*args, **kwargs)  # Usually None

            if hud_task_run_id:
                create_notification_record(
                    **record_data_base,
                    status=StatusType.COMPLETED,
                    start_time=start_time,
                    end_time=time.time(),
                    duration=time.time() - start_time,
                )
            return result
        except Exception as e:
            if hud_task_run_id:
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
            logger.info("Manual test recorded: %s", record.model_dump(exclude_none=True))
            return True
        except Exception as e:
            logger.warning("Manual test not recorded: %s", e)
            return False
