from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

from hud.telemetry.context import get_current_task_run_id as actual_get_current_task_run_id
from hud.telemetry.context import is_root_trace as actual_is_root_trace
from hud.telemetry.context import reset_context
from hud.telemetry.context import set_current_task_run_id as actual_set_current_task_run_id
from hud.telemetry.trace import (
    init_telemetry,
    register_trace,
    trace,
)


@pytest.fixture(autouse=True)
def reset_telemetry_context_fixture():
    """Ensures telemetry context is reset before and after each test in this file."""
    reset_context()
    yield
    reset_context()


class TestInitTelemetry:
    """Test telemetry initialization."""

    @patch("hud.telemetry.trace.registry")
    def test_init_telemetry(self, mock_registry):
        """Test telemetry initialization calls registry.install_all."""
        init_telemetry()
        mock_registry.install_all.assert_called_once()


class TestTrace:
    """Test the trace context manager."""

    @patch("hud.telemetry.trace.submit_to_worker_loop")
    @patch("hud.telemetry.trace.flush_buffer")
    def test_trace_basic(self, mock_flush, mock_submit_loop):
        """Test basic trace functionality and context setting."""
        mock_flush.return_value = []
        mock_submit_loop.return_value = MagicMock()

        initial_root_state = actual_is_root_trace.get()

        with trace() as task_run_id:
            assert isinstance(task_run_id, str)
            uuid.UUID(task_run_id)
            assert actual_get_current_task_run_id() == task_run_id
            assert actual_is_root_trace.get() is True

        assert actual_get_current_task_run_id() is None
        assert actual_is_root_trace.get() == initial_root_state
        mock_flush.assert_called_once()
        mock_submit_loop.assert_not_called()

    @patch("hud.telemetry.trace.submit_to_worker_loop")
    @patch("hud.telemetry.trace.flush_buffer")
    def test_trace_with_name_and_attributes(self, mock_flush, mock_submit_loop):
        """Test trace with name and attributes, checking they are passed on."""
        mock_mcp_calls = [MagicMock()]
        mock_flush.return_value = mock_mcp_calls
        mock_submit_loop.return_value = MagicMock()

        trace_name = "test_trace_with_data"
        attrs = {"key": "value", "number": 42}

        with trace(name=trace_name, attributes=attrs) as task_run_id:
            assert isinstance(task_run_id, str)

        mock_flush.assert_called_once()
        mock_submit_loop.assert_called_once()

    @patch("hud.telemetry.trace.submit_to_worker_loop")
    @patch("hud.telemetry.trace.flush_buffer")
    @patch("hud.telemetry.trace.export_telemetry_coro")
    def test_trace_with_mcp_calls_exports(
        self,
        mock_export_actual_coro,
        mock_flush,
        mock_submit_loop,
    ):
        """Test trace with MCP calls exports telemetry with correct data."""
        mock_mcp_calls = [MagicMock(), MagicMock()]
        mock_flush.return_value = mock_mcp_calls
        mock_submit_loop.return_value = MagicMock()

        test_attrs = {"custom_attr": "test_val"}
        test_name = "mcp_export_test"

        with trace(name=test_name, attributes=test_attrs) as task_run_id:
            pass

        mock_flush.assert_called_once()
        mock_submit_loop.assert_called_once()

        mock_export_actual_coro.assert_called_once()
        args, kwargs = mock_export_actual_coro.call_args
        assert kwargs["task_run_id"] == task_run_id
        assert kwargs["mcp_calls"] == mock_mcp_calls
        assert kwargs["trace_attributes"]["trace_name"] == test_name
        assert kwargs["trace_attributes"]["custom_attr"] == "test_val"
        assert "start_time" in kwargs["trace_attributes"]
        assert "end_time" in kwargs["trace_attributes"]
        assert "duration" in kwargs["trace_attributes"]
        assert kwargs["trace_attributes"]["is_root"] is True

    @patch("hud.telemetry.trace.submit_to_worker_loop")
    @patch("hud.telemetry.trace.flush_buffer")
    def test_trace_nested(self, mock_flush_internal, mock_submit_loop_internal):
        """Test nested traces, verifying context restoration and root trace logic."""
        actual_set_current_task_run_id(None)
        actual_is_root_trace.set(False)

        mock_flush_internal.return_value = []
        mock_submit_loop_internal.return_value = MagicMock()

        assert actual_get_current_task_run_id() is None
        assert actual_is_root_trace.get() is False

        with trace(name="outer") as outer_id:
            assert actual_get_current_task_run_id() == outer_id
            assert actual_is_root_trace.get() is True
            with trace(name="inner") as inner_id:
                assert actual_get_current_task_run_id() == inner_id
                assert actual_is_root_trace.get() is False
                assert outer_id != inner_id
            assert actual_get_current_task_run_id() == outer_id
            assert actual_is_root_trace.get() is True

        assert actual_get_current_task_run_id() is None
        assert actual_is_root_trace.get() is False
        assert mock_flush_internal.call_count == 2
        mock_submit_loop_internal.assert_not_called()

    @patch("hud.telemetry.trace.submit_to_worker_loop")
    @patch("hud.telemetry.trace.flush_buffer")
    def test_trace_exception_handling(self, mock_flush, mock_submit_loop):
        """Test trace handles exceptions properly and restores context."""
        initial_task_id_before_trace = "pre_existing_id_123"
        initial_root_state_before_trace = True
        actual_set_current_task_run_id(initial_task_id_before_trace)
        actual_is_root_trace.set(initial_root_state_before_trace)

        mock_flush.return_value = []
        mock_submit_loop.return_value = MagicMock()

        with pytest.raises(ValueError, match="Test exception"), trace(name="trace_with_exception"):
            assert actual_get_current_task_run_id() != initial_task_id_before_trace
            assert actual_is_root_trace.get() is False
            raise ValueError("Test exception")

        mock_flush.assert_called_once()
        assert actual_get_current_task_run_id() == initial_task_id_before_trace
        assert actual_is_root_trace.get() == initial_root_state_before_trace
        mock_submit_loop.assert_not_called()


class TestRegisterTrace:
    """Test the register_trace decorator."""

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_sync_function(self, mock_trace_context_manager):
        mock_trace_context_manager.return_value.__enter__.return_value = "mocked_task_id"
        mock_trace_context_manager.return_value.__exit__.return_value = None

        @register_trace(name="test_func_sync")
        def sync_function(x, y):
            return x + y

        result = sync_function(1, 2)
        assert result == 3
        mock_trace_context_manager.assert_called_once_with(name="test_func_sync", attributes=None)

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_async_function(self, mock_trace_context_manager):
        mock_trace_context_manager.return_value.__enter__.return_value = "mocked_task_id"
        mock_trace_context_manager.return_value.__exit__.return_value = None

        @register_trace(name="test_func_async")
        async def async_function(x, y):
            return x + y

        async def run_test():
            result = await async_function(1, 2)
            assert result == 3
            mock_trace_context_manager.assert_called_once_with(
                name="test_func_async", attributes=None
            )

        asyncio.run(run_test())

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_with_attributes(self, mock_trace_context_manager):
        """Test register_trace with attributes."""
        mock_trace_context_manager.return_value.__enter__.return_value = "task_id"
        mock_trace_context_manager.return_value.__exit__.return_value = None

        attrs = {"operation": "add"}

        @register_trace(name="test_func", attributes=attrs)
        def func_with_attrs(x):
            return x * 2

        result = func_with_attrs(5)
        assert result == 10
        mock_trace_context_manager.assert_called_once_with(name="test_func", attributes=attrs)

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_without_name(self, mock_trace_context_manager):
        """Test register_trace uses function name when name not provided."""
        mock_trace_context_manager.return_value.__enter__.return_value = "task_id"
        mock_trace_context_manager.return_value.__exit__.return_value = None

        @register_trace()
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"
        mock_trace_context_manager.assert_called_once_with(name="my_function", attributes=None)

    def test_register_trace_preserves_function_metadata(self):
        """Test register_trace preserves original function metadata."""

        @register_trace(name="test")
        def original_function():
            """Original docstring."""

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring."

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_exception_propagation(self, mock_trace_context_manager):
        """Test register_trace propagates exceptions."""
        mock_trace_context_manager.return_value.__enter__.return_value = "task_id"
        mock_trace_context_manager.return_value.__exit__.return_value = None

        @register_trace()
        def failing_function():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            failing_function()

        mock_trace_context_manager.assert_called_once()
