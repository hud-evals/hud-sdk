from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

# Import MCP types
from mcp.types import JSONRPCError, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from mcp.shared.message import SessionMessage


class DirectionType(str, Enum):
    """Direction of an MCP message"""

    SENT = "sent"
    RECEIVED = "received"


class StatusType(str, Enum):
    """Status of an MCP operation"""

    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"


class MCPCallType(str, Enum):
    """Known MCP call types"""

    SEND_REQUEST = "mcp.shared.session.send_request"
    SEND_NOTIFICATION = "mcp.shared.session.send_notification"
    RECEIVE_RESPONSE = "mcp.shared.session.receive_response"
    RECEIVE_REQUEST = "mcp.shared.session.receive_request"
    STREAM_READ = "mcp.stream.read"
    STREAM_WRITE = "mcp.stream.write"
    HANDLE_INCOMING = "mcp.handle_incoming"
    MANUAL_TEST = "manual.test"


class BaseMCPCall(BaseModel):
    """Base model for all MCP telemetry records"""

    task_run_id: str
    call_type: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    method: str = "unknown_method"
    status: StatusType
    direction: DirectionType | None = None
    # Additional data that might be useful for any call
    message_id: str | int | None = None

    # Mapping of call types to model classes - to be populated by subclasses
    _call_type_mapping: ClassVar[dict[str, type["BaseMCPCall"]]] = {}

    @field_validator("call_type")
    @classmethod
    def validate_call_type(cls, v: str) -> str:
        """Allow any string but preferably from MCPCallType"""
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseMCPCall:
        """Create a record from a dictionary, using the appropriate subclass"""
        call_type = data.get("call_type", "")
        record_cls = cls._call_type_mapping.get(call_type, BaseMCPCall)
        return record_cls.model_validate(data)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses in the mapping by their default call_type"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__annotations__") and "call_type" in cls.__annotations__:
            default_call_type = getattr(cls, "call_type", None)
            if isinstance(default_call_type, str):
                BaseMCPCall._call_type_mapping[default_call_type] = cls


class MCPRequestCall(BaseMCPCall):
    """Record for an MCP request"""

    direction: DirectionType = DirectionType.SENT
    call_type: str = MCPCallType.SEND_REQUEST
    start_time: float
    end_time: float | None = None
    duration: float | None = None
    request_id: str | int | None = None
    request_data: dict[str, Any] | None = None

    @classmethod
    def from_jsonrpc_request(
        cls,
        request: JSONRPCRequest,
        task_run_id: str,
        status: StatusType = StatusType.STARTED,
        **kwargs: Any,
    ) -> MCPRequestCall:
        """Create telemetry record from a JSONRPCRequest"""
        return cls(
            task_run_id=task_run_id,
            status=status,
            request_id=request.id,
            message_id=request.id,
            method=request.method,
            request_data=request.model_dump(exclude_none=True),
            start_time=datetime.now().timestamp(),
            **kwargs,
        )

    @classmethod
    def from_session_message(
        cls,
        message: SessionMessage,
        task_run_id: str,
        status: StatusType = StatusType.STARTED,
        **kwargs: Any,
    ) -> MCPRequestCall | None:
        """Create telemetry record from a SessionMessage containing a JSONRPCRequest"""
        if (
            hasattr(message, "message")
            and hasattr(message.message, "root")
            and isinstance(message.message.root, JSONRPCRequest)
        ):
            return cls.from_jsonrpc_request(
                message.message.root, task_run_id=task_run_id, status=status, **kwargs
            )
        return None


class MCPResponseCall(BaseMCPCall):
    """Record for an MCP response"""

    direction: DirectionType = DirectionType.RECEIVED
    call_type: str = MCPCallType.RECEIVE_RESPONSE
    is_response_or_error: bool = True
    is_error: bool = False
    response_id: str | int | None = None
    related_request_id: str | int | None = None
    response_data: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None

    @classmethod
    def from_jsonrpc_response(
        cls, response: JSONRPCResponse | JSONRPCError, task_run_id: str, **kwargs: Any
    ) -> MCPResponseCall:
        """Create telemetry record from a JSONRPCResponse or JSONRPCError"""
        is_error = isinstance(response, JSONRPCError)

        result = cls(
            task_run_id=task_run_id,
            status=StatusType.COMPLETED,
            response_id=response.id,
            message_id=response.id,
            related_request_id=response.id,  # In MCP, response ID matches request ID
            is_error=is_error,
            method=f"response_to_id_{response.id}",
            response_data=response.model_dump(exclude_none=True),
            **kwargs,
        )

        if is_error and hasattr(response, "error"):
            result.error = response.error.message
            result.error_type = str(response.error.code)

        return result

    @classmethod
    def from_session_message(
        cls, message: SessionMessage, task_run_id: str, **kwargs: Any
    ) -> MCPResponseCall | None:
        """Create telemetry record from a SessionMessage containing a response or error"""
        if (
            hasattr(message, "message")
            and hasattr(message.message, "root")
            and isinstance(message.message.root, JSONRPCResponse | JSONRPCError)
        ):
            return cls.from_jsonrpc_response(
                message.message.root, task_run_id=task_run_id, **kwargs
            )
        return None


class MCPNotificationCall(BaseMCPCall):
    """Record for an MCP notification"""

    direction: DirectionType = DirectionType.SENT
    call_type: str = MCPCallType.SEND_NOTIFICATION
    start_time: float
    end_time: float | None = None
    duration: float | None = None
    notification_data: dict[str, Any] | None = None

    @classmethod
    def from_jsonrpc_notification(
        cls,
        notification: JSONRPCNotification,
        task_run_id: str,
        status: StatusType = StatusType.STARTED,
        **kwargs: Any,
    ) -> MCPNotificationCall:
        """Create telemetry record from a JSONRPCNotification"""
        return cls(
            task_run_id=task_run_id,
            status=status,
            method=notification.method,
            notification_data=notification.model_dump(exclude_none=True),
            start_time=datetime.now().timestamp(),
            **kwargs,
        )

    @classmethod
    def from_session_message(
        cls,
        message: SessionMessage,
        task_run_id: str,
        status: StatusType = StatusType.STARTED,
        **kwargs: Any,
    ) -> MCPNotificationCall | None:
        """Create telemetry record from a SessionMessage containing a JSONRPCNotification"""
        if (
            hasattr(message, "message")
            and hasattr(message.message, "root")
            and isinstance(message.message.root, JSONRPCNotification)
        ):
            return cls.from_jsonrpc_notification(
                message.message.root, task_run_id=task_run_id, status=status, **kwargs
            )
        return None


class MCPStreamEvent(BaseMCPCall):
    """Record for an MCP stream event (read or write)"""

    stream_event: bool = True
    event_type: str = Field(..., description="Type of stream event: read or write")
    item_type: str | None = None
    is_response_or_error: bool = False
    message_data: dict[str, Any] | None = None

    @classmethod
    def from_session_message(
        cls, message: SessionMessage, task_run_id: str, event_type: str, **kwargs: Any
    ) -> MCPStreamEvent:
        """Create telemetry record for a stream event"""
        method_name = "unknown_stream_operation"
        is_response = False
        item_type = "unknown"
        message_data = None

        if hasattr(message, "message") and hasattr(message.message, "root"):
            msg_root = message.message.root
            item_type = type(msg_root).__name__
            message_data = msg_root.model_dump(exclude_none=True)

            # Check type first before accessing attributes
            if isinstance(msg_root, JSONRPCRequest | JSONRPCNotification) and hasattr(
                msg_root, "method"
            ):
                method_name = msg_root.method
            elif isinstance(msg_root, JSONRPCResponse | JSONRPCError) and hasattr(msg_root, "id"):
                method_name = f"response_to_id_{msg_root.id}"
                is_response = True

        return cls(
            task_run_id=task_run_id,
            status=StatusType.COMPLETED,
            method=method_name,
            event_type=event_type,
            item_type=item_type,
            is_response_or_error=is_response,
            message_data=message_data,
            timestamp=datetime.now().timestamp(),
            **kwargs,
        )


class MCPManualTestCall(BaseMCPCall):
    """Record for a manual test record"""

    call_type: str = MCPCallType.MANUAL_TEST
    custom_data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(cls, task_run_id: str, **custom_data: Any) -> MCPManualTestCall:
        """Create a manual test record with custom data"""
        return cls(
            task_run_id=task_run_id,
            status=StatusType.COMPLETED,
            custom_data=custom_data,
            timestamp=datetime.now().timestamp(),
        )


class MCPTelemetryRecord(BaseModel):
    """Container for a set of related MCP telemetry records"""

    task_run_id: str
    records: list[BaseMCPCall]
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    @property
    def count_by_type(self) -> dict[str, int]:
        """Count records by call_type"""
        result: dict[str, int] = {}
        for record in self.records:
            result[record.call_type] = result.get(record.call_type, 0) + 1
        return result

    @property
    def count_by_direction(self) -> dict[str, int]:
        """Count records by direction"""
        result: dict[str, int] = {}
        for record in self.records:
            if record.direction:
                direction = record.direction.value
                result[direction] = result.get(direction, 0) + 1
        return result


class TrajectoryStep(BaseModel):
    """Model representing a single step in a trajectory, for export."""

    type: str = Field(default="mcp-step")  # Default for MCP calls
    observation_url: str | None = None
    observation_text: str | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)
    start_timestamp: str | None = None  # ISO 8601 format
    end_timestamp: str | None = None  # ISO 8601 format
    metadata: dict[str, Any] = Field(default_factory=dict)
