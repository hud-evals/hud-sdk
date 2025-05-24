from __future__ import annotations

import pytest

from hud.adapters.claude import ClaudeAdapter
from hud.adapters.common.types import (
    ClickAction,
    DragAction,
    MoveAction,
    PositionFetch,
    PressAction,
    ResponseAction,
    ScreenshotFetch,
    ScrollAction,
    TypeAction,
    WaitAction,
)


class TestClaudeAdapter:
    """Test the ClaudeAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Fixture providing a clean adapter instance."""
        return ClaudeAdapter()

    def test_init(self, adapter):
        """Test adapter initialization."""
        assert adapter.agent_width == 1024
        assert adapter.agent_height == 768
        assert adapter.env_width == 1920  # Inherited from parent
        assert adapter.env_height == 1080  # Inherited from parent

    def test_key_map_constants(self, adapter):
        """Test KEY_MAP constants."""
        assert adapter.KEY_MAP["return"] == "enter"
        assert adapter.KEY_MAP["super"] == "win"
        assert adapter.KEY_MAP["super_l"] == "win"
        assert adapter.KEY_MAP["super_r"] == "win"
        assert adapter.KEY_MAP["right shift"] == "shift"
        assert adapter.KEY_MAP["left shift"] == "shift"

    def test_map_key_mapped(self, adapter):
        """Test _map_key with mapped keys."""
        assert adapter._map_key("return") == "enter"
        assert adapter._map_key("RETURN") == "enter"  # Test case insensitive
        assert adapter._map_key("super") == "win"
        assert adapter._map_key("Super_L") == "win"

    def test_map_key_unmapped(self, adapter):
        """Test _map_key with unmapped keys."""
        assert adapter._map_key("space") == "space"
        assert adapter._map_key("CTRL") == "ctrl"
        assert adapter._map_key("Unknown") == "unknown"


class TestClaudeAdapterConvert:
    """Test the convert method of ClaudeAdapter."""

    @pytest.fixture
    def adapter(self):
        """Fixture providing a clean adapter instance."""
        return ClaudeAdapter()

    def test_convert_key_single(self, adapter):
        """Test converting single key action."""
        data = {"action": "key", "text": "space"}
        result = adapter.convert(data)

        assert isinstance(result, PressAction)
        assert result.keys == ["space"]

    def test_convert_key_mapped(self, adapter):
        """Test converting mapped key action."""
        data = {"action": "key", "text": "return"}
        result = adapter.convert(data)

        assert isinstance(result, PressAction)
        assert result.keys == ["enter"]

    def test_convert_key_combination(self, adapter):
        """Test converting key combination action."""
        data = {"action": "key", "text": "ctrl+c"}
        result = adapter.convert(data)

        assert isinstance(result, PressAction)
        assert result.keys == ["ctrl", "c"]

    def test_convert_key_combination_mapped(self, adapter):
        """Test converting key combination with mapped keys."""
        data = {"action": "key", "text": "super+return"}
        result = adapter.convert(data)

        assert isinstance(result, PressAction)
        assert result.keys == ["win", "enter"]

    def test_convert_key_missing_text(self, adapter):
        """Test converting key action with missing text."""
        data = {"action": "key"}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Invalid action" in str(exc_info.value)

    def test_convert_type_action(self, adapter):
        """Test converting type action."""
        data = {"action": "type", "text": "Hello, World!"}
        result = adapter.convert(data)

        assert isinstance(result, TypeAction)
        assert result.text == "Hello, World!"
        assert result.enter_after is False

    def test_convert_type_missing_text(self, adapter):
        """Test converting type action with missing text."""
        data = {"action": "type"}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Invalid action" in str(exc_info.value)

    def test_convert_mouse_move(self, adapter):
        """Test converting mouse move action."""
        data = {"action": "mouse_move", "coordinate": [100, 200]}
        result = adapter.convert(data)

        assert isinstance(result, MoveAction)
        assert result.point is not None
        assert result.point.x == 100
        assert result.point.y == 200

    def test_convert_mouse_move_invalid_coordinate(self, adapter):
        """Test converting mouse move with invalid coordinate."""
        # Wrong number of coordinates
        data = {"action": "mouse_move", "coordinate": [100]}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Invalid action" in str(exc_info.value)

    def test_convert_mouse_move_missing_coordinate(self, adapter):
        """Test converting mouse move with missing coordinate."""
        data = {"action": "mouse_move"}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Invalid action" in str(exc_info.value)

    def test_convert_left_click(self, adapter):
        """Test converting left click action."""
        data = {"action": "left_click", "coordinate": [150, 250]}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 150
        assert result.point.y == 250
        assert result.button == "left"

    def test_convert_right_click(self, adapter):
        """Test converting right click action."""
        data = {"action": "right_click", "coordinate": [300, 400]}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 300
        assert result.point.y == 400
        assert result.button == "right"

    def test_convert_middle_click(self, adapter):
        """Test converting middle click action."""
        data = {"action": "middle_click", "coordinate": [350, 450]}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 350
        assert result.point.y == 450
        assert result.button == "middle"

    def test_convert_double_click(self, adapter):
        """Test converting double click action."""
        data = {"action": "double_click", "coordinate": [200, 300]}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 200
        assert result.point.y == 300
        assert result.button == "left"
        assert result.pattern == [100]

    def test_convert_triple_click(self, adapter):
        """Test converting triple click action."""
        data = {"action": "triple_click", "coordinate": [250, 350]}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 250
        assert result.point.y == 350
        assert result.button == "left"
        assert result.pattern == [100, 100]

    def test_convert_left_click_drag_with_move_history(self, adapter):
        """Test converting left click drag with move action in history."""
        # First add a move action to memory
        move_data = {"action": "mouse_move", "coordinate": [100, 200]}
        adapter.adapt(move_data)

        # Now test drag
        drag_data = {"action": "left_click_drag", "coordinate": [300, 400]}
        result = adapter.convert(drag_data)

        assert isinstance(result, DragAction)
        assert len(result.path) == 2
        assert result.path[0].x == 100
        assert result.path[0].y == 200
        assert result.path[1].x == 300
        assert result.path[1].y == 400

    def test_convert_left_click_drag_with_click_history(self, adapter):
        """Test converting left click drag with click action in history."""
        # First add a click action to memory
        click_data = {"action": "left_click", "coordinate": [150, 250]}
        adapter.adapt(click_data)

        # Now test drag
        drag_data = {"action": "left_click_drag", "coordinate": [350, 450]}
        result = adapter.convert(drag_data)

        assert isinstance(result, DragAction)
        assert len(result.path) == 2
        assert result.path[0].x == 150
        assert result.path[0].y == 250
        assert result.path[1].x == 350
        assert result.path[1].y == 450

    def test_convert_left_click_drag_without_history(self, adapter):
        """Test converting left click drag without proper history."""
        data = {"action": "left_click_drag", "coordinate": [300, 400]}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Left click drag must be preceded by a move or click action" in str(exc_info.value)

    def test_convert_left_click_drag_with_invalid_history(self, adapter):
        """Test converting left click drag with invalid history."""
        # Add a type action (not move or click) to memory
        type_data = {"action": "type", "text": "hello"}
        adapter.adapt(type_data)

        # Now test drag should fail
        drag_data = {"action": "left_click_drag", "coordinate": [300, 400]}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(drag_data)

        assert "Left click drag must be preceded by a move or click action" in str(exc_info.value)

    def test_convert_scroll_up(self, adapter):
        """Test converting scroll up action."""
        data = {
            "action": "scroll",
            "coordinate": [500, 600],
            "scroll_direction": "up",
            "scroll_amount": 3,
        }
        result = adapter.convert(data)

        assert isinstance(result, ScrollAction)
        assert result.point is not None
        assert result.scroll is not None
        assert result.point.x == 500
        assert result.point.y == 600
        assert result.scroll.x == 0
        assert result.scroll.y == -3

    def test_convert_scroll_down(self, adapter):
        """Test converting scroll down action."""
        data = {
            "action": "scroll",
            "coordinate": [500, 600],
            "scroll_direction": "down",
            "scroll_amount": 5,
        }
        result = adapter.convert(data)

        assert isinstance(result, ScrollAction)
        assert result.point is not None
        assert result.scroll is not None
        assert result.point.x == 500
        assert result.point.y == 600
        assert result.scroll.x == 0
        assert result.scroll.y == 5

    def test_convert_scroll_left(self, adapter):
        """Test converting scroll left action."""
        data = {
            "action": "scroll",
            "coordinate": [500, 600],
            "scroll_direction": "left",
            "scroll_amount": 2,
        }
        result = adapter.convert(data)

        assert isinstance(result, ScrollAction)
        assert result.point is not None
        assert result.scroll is not None
        assert result.point.x == 500
        assert result.point.y == 600
        assert result.scroll.x == -2
        assert result.scroll.y == 0

    def test_convert_scroll_right(self, adapter):
        """Test converting scroll right action."""
        data = {
            "action": "scroll",
            "coordinate": [500, 600],
            "scroll_direction": "right",
            "scroll_amount": 4,
        }
        result = adapter.convert(data)

        assert isinstance(result, ScrollAction)
        assert result.point is not None
        assert result.scroll is not None
        assert result.point.x == 500
        assert result.point.y == 600
        assert result.scroll.x == 4
        assert result.scroll.y == 0

    def test_convert_scroll_invalid_direction(self, adapter):
        """Test converting scroll with invalid direction."""
        data = {
            "action": "scroll",
            "coordinate": [500, 600],
            "scroll_direction": "diagonal",
            "scroll_amount": 3,
        }

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Unsupported scroll direction: diagonal" in str(exc_info.value)

    def test_convert_scroll_missing_direction(self, adapter):
        """Test converting scroll with missing direction."""
        data = {"action": "scroll", "coordinate": [500, 600], "scroll_amount": 3}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Invalid action" in str(exc_info.value)

    def test_convert_screenshot(self, adapter):
        """Test converting screenshot action."""
        data = {"action": "screenshot"}
        result = adapter.convert(data)

        assert isinstance(result, ScreenshotFetch)

    def test_convert_cursor_position(self, adapter):
        """Test converting cursor position action."""
        data = {"action": "cursor_position"}
        result = adapter.convert(data)

        assert isinstance(result, PositionFetch)

    def test_convert_wait(self, adapter):
        """Test converting wait action."""
        data = {"action": "wait", "duration": 2500}
        result = adapter.convert(data)

        assert isinstance(result, WaitAction)
        assert result.time == 2500

    def test_convert_wait_missing_duration(self, adapter):
        """Test converting wait action with missing duration."""
        data = {"action": "wait"}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Invalid action" in str(exc_info.value)

    def test_convert_response(self, adapter):
        """Test converting response action."""
        data = {"action": "response", "text": "Task completed successfully"}
        result = adapter.convert(data)

        assert isinstance(result, ResponseAction)
        assert result.text == "Task completed successfully"

    def test_convert_response_default_text(self, adapter):
        """Test converting response action with default text."""
        data = {"action": "response"}
        result = adapter.convert(data)

        assert isinstance(result, ResponseAction)
        assert result.text == ""

    def test_convert_unsupported_action(self, adapter):
        """Test converting unsupported action type."""
        data = {"action": "unsupported_action"}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Unsupported action type: unsupported_action" in str(exc_info.value)

    def test_convert_missing_action_field(self, adapter):
        """Test converting data without action field."""
        data = {"text": "hello"}  # Missing action

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Unsupported action type: None" in str(exc_info.value)

    def test_convert_invalid_data_structure(self, adapter):
        """Test converting invalid data structure."""
        with pytest.raises(ValueError) as exc_info:
            adapter.convert("invalid_data")

        assert "Invalid action" in str(exc_info.value)

    def test_convert_none_data(self, adapter):
        """Test converting None data."""
        with pytest.raises(ValueError) as exc_info:
            adapter.convert(None)

        assert "Invalid action" in str(exc_info.value)


class TestClaudeAdapterIntegration:
    """Integration tests for ClaudeAdapter."""

    @pytest.fixture
    def adapter(self):
        """Fixture providing a clean adapter instance."""
        return ClaudeAdapter()

    def test_full_click_pipeline(self, adapter):
        """Test full click action processing pipeline."""
        # Set adapter dimensions to avoid scaling
        adapter.agent_width = 1920
        adapter.agent_height = 1080
        adapter.env_width = 1920
        adapter.env_height = 1080

        raw_action = {"action": "left_click", "coordinate": [100, 200]}

        result = adapter.adapt(raw_action)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 100
        assert result.point.y == 200
        assert result.button == "left"

        # Check that it was added to memory
        assert len(adapter.memory) == 1
        assert adapter.memory[0] == result

    def test_drag_sequence(self, adapter):
        """Test complete drag sequence."""
        # Set adapter dimensions to avoid scaling
        adapter.agent_width = 1920
        adapter.agent_height = 1080
        adapter.env_width = 1920
        adapter.env_height = 1080

        # First move to start position
        move_action = {"action": "mouse_move", "coordinate": [100, 200]}
        move_result = adapter.adapt(move_action)

        # Then drag to end position
        drag_action = {"action": "left_click_drag", "coordinate": [300, 400]}
        drag_result = adapter.adapt(drag_action)

        assert isinstance(move_result, MoveAction)
        assert isinstance(drag_result, DragAction)
        assert len(drag_result.path) == 2
        assert drag_result.path[0] == move_result.point

        # Check memory contains both actions
        assert len(adapter.memory) == 2

    def test_complex_action_sequence(self, adapter):
        """Test complex sequence of different actions."""
        actions = [
            {"action": "mouse_move", "coordinate": [100, 200]},
            {"action": "left_click", "coordinate": [150, 250]},
            {"action": "type", "text": "Hello"},
            {"action": "key", "text": "ctrl+a"},
            {"action": "wait", "duration": 1000},
            {"action": "screenshot"},
        ]

        results = adapter.adapt_list(actions)

        assert len(results) == 6
        assert isinstance(results[0], MoveAction)
        assert isinstance(results[1], ClickAction)
        assert isinstance(results[2], TypeAction)
        assert isinstance(results[3], PressAction)
        assert isinstance(results[4], WaitAction)
        assert isinstance(results[5], ScreenshotFetch)

        # Check memory
        assert len(adapter.memory) == 6
