from __future__ import annotations

from unittest.mock import MagicMock

from hud.telemetry.context import (
    buffer_mcp_call,
    flush_buffer,
    get_current_task_run_id,
    is_root_trace,
    set_current_task_run_id,
)
from hud.telemetry.mcp_models import BaseMCPCall


class TestTaskRunIdContext:
    """Test task run ID context management."""

    def test_get_current_task_run_id_initial(self):
        """Test getting task run ID when none is set."""
        # Reset context for clean test
        set_current_task_run_id(None)
        result = get_current_task_run_id()
        assert result is None

    def test_set_and_get_task_run_id(self):
        """Test setting and getting task run ID."""
        test_id = "test-task-run-id"
        set_current_task_run_id(test_id)
        result = get_current_task_run_id()
        assert result == test_id

    def test_task_run_id_isolation(self):
        """Test that task run IDs are isolated per context."""
        # This test simulates what would happen in different contexts
        set_current_task_run_id("context-1")
        assert get_current_task_run_id() == "context-1"

        set_current_task_run_id("context-2")
        assert get_current_task_run_id() == "context-2"

        # Reset to None
        set_current_task_run_id(None)
        assert get_current_task_run_id() is None


class TestRootTraceContext:
    """Test root trace context management."""

    def test_is_root_trace_initial(self):
        """Test is_root_trace initial state."""
        # The initial state may vary, so we just test that it returns a boolean
        result = is_root_trace.get()
        assert isinstance(result, bool)

    def test_set_root_trace(self):
        """Test setting root trace state."""
        is_root_trace.set(True)
        assert is_root_trace.get() is True

        is_root_trace.set(False)
        assert is_root_trace.get() is False


class TestMCPCallBuffer:
    """Test MCP call buffer management."""

    def setUp(self):
        """Clear buffer before each test."""
        # Flush any existing calls and reset context
        flush_buffer()
        set_current_task_run_id(None)

    def test_flush_buffer_empty(self):
        """Test flushing empty buffer."""
        self.setUp()
        result = flush_buffer()
        assert result == []

    def test_add_and_flush_mcp_call(self):
        """Test adding and flushing MCP calls."""
        self.setUp()

        # Set active task run ID
        set_current_task_run_id("test-task")

        # Create mock MCP call with required attributes
        mock_call = MagicMock(spec=BaseMCPCall)
        mock_call.model_dump.return_value = {"type": "test", "task_run_id": "test-task"}
        mock_call.task_run_id = "test-task"

        buffer_mcp_call(mock_call)

        # Flush should return the call and clear buffer
        result = flush_buffer()
        assert len(result) == 1
        assert result[0] == mock_call

        # Buffer should be empty after flush
        result2 = flush_buffer()
        assert result2 == []

    def test_add_multiple_mcp_calls(self):
        """Test adding multiple MCP calls."""
        self.setUp()

        # Set active task run ID
        set_current_task_run_id("test-task")

        # Create multiple mock calls
        mock_calls = []
        for i in range(3):
            mock_call = MagicMock(spec=BaseMCPCall)
            mock_call.model_dump.return_value = {"type": f"test_{i}", "task_run_id": "test-task"}
            mock_call.task_run_id = "test-task"
            mock_calls.append(mock_call)
            buffer_mcp_call(mock_call)

        # Flush should return all calls
        result = flush_buffer()
        assert len(result) == 3
        assert result == mock_calls

    def test_buffer_isolation_per_task(self):
        """Test that MCP call buffers contain all calls regardless of task ID."""
        self.setUp()

        # Set task run ID 1
        set_current_task_run_id("task-1")
        mock_call_1 = MagicMock(spec=BaseMCPCall)
        mock_call_1.task_run_id = "task-1"
        mock_call_1.model_dump.return_value = {"type": "test", "task_run_id": "task-1"}
        buffer_mcp_call(mock_call_1)

        # Set task run ID 2
        set_current_task_run_id("task-2")
        mock_call_2 = MagicMock(spec=BaseMCPCall)
        mock_call_2.task_run_id = "task-2"
        mock_call_2.model_dump.return_value = {"type": "test", "task_run_id": "task-2"}
        buffer_mcp_call(mock_call_2)

        # Flush should return all calls from both tasks
        result = flush_buffer()
        assert len(result) == 1
        assert result[0] == mock_call_2

        set_current_task_run_id("task-1")
        result2 = flush_buffer()
        assert len(result2) == 1
        assert result2[0] == mock_call_1

    def test_buffer_mcp_call_without_task_id(self):
        """Test adding MCP call when no task run ID is set."""
        self.setUp()
        set_current_task_run_id(None)

        mock_call = MagicMock(spec=BaseMCPCall)
        mock_call.task_run_id = None
        buffer_mcp_call(mock_call)

        # Should not buffer anything when no task ID is set
        result = flush_buffer()
        assert len(result) == 0


class TestContextIntegration:
    """Integration tests for context management."""

    def test_context_lifecycle(self):
        """Test complete context lifecycle."""
        # Start with clean state
        set_current_task_run_id(None)
        flush_buffer()
        is_root_trace.set(False)

        # Set up trace context
        task_id = "integration-test-task"
        set_current_task_run_id(task_id)
        is_root_trace.set(True)

        # Add some MCP calls
        mock_calls = []
        for i in range(2):
            mock_call = MagicMock(spec=BaseMCPCall)
            mock_call.model_dump.return_value = {
                "type": f"integration_test_{i}",
                "task_run_id": task_id,
            }
            mock_call.task_run_id = task_id
            mock_calls.append(mock_call)
            buffer_mcp_call(mock_call)

        # Verify context state
        assert get_current_task_run_id() == task_id
        assert is_root_trace.get() is True

        # Flush and verify
        result = flush_buffer()
        assert len(result) == 2
        assert result == mock_calls

        # Clean up
        set_current_task_run_id(None)
        is_root_trace.set(False)

        # Verify cleanup
        assert get_current_task_run_id() is None
        assert flush_buffer() == []
