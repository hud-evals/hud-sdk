from __future__ import annotations

import pytest

from hud.adapters.common.types import (
    ClickAction,
    DragAction,
    MoveAction,
    PressAction,
    ResponseAction,
    ScreenshotFetch,
    ScrollAction,
    TypeAction,
    WaitAction,
)
from hud.adapters.operator import OperatorAdapter


class TestOperatorAdapter:
    """Test the OperatorAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Fixture providing a clean adapter instance."""
        return OperatorAdapter()

    def test_init(self, adapter):
        """Test adapter initialization."""
        assert adapter.agent_width == 1024
        assert adapter.agent_height == 768
        assert adapter.env_width == 1920  # Inherited from parent
        assert adapter.env_height == 1080  # Inherited from parent

    def test_key_map_constants(self, adapter):
        """Test KEY_MAP constants."""
        assert adapter.KEY_MAP["return"] == "enter"
        assert adapter.KEY_MAP["arrowup"] == "up"
        assert adapter.KEY_MAP["arrowdown"] == "down"
        assert adapter.KEY_MAP["arrowleft"] == "left"
        assert adapter.KEY_MAP["arrowright"] == "right"

    def test_button_map_constants(self, adapter):
        """Test BUTTON_MAP constants."""
        assert adapter.BUTTON_MAP["wheel"] == "middle"

    def test_map_key_mapped(self, adapter):
        """Test _map_key with mapped keys."""
        assert adapter._map_key("return") == "enter"
        assert adapter._map_key("RETURN") == "enter"  # Test case insensitive
        assert adapter._map_key("arrowup") == "up"
        assert adapter._map_key("ArrowDown") == "down"

    def test_map_key_unmapped(self, adapter):
        """Test _map_key with unmapped keys."""
        assert adapter._map_key("space") == "space"
        assert adapter._map_key("CTRL") == "ctrl"
        assert adapter._map_key("Unknown") == "unknown"


class TestOperatorAdapterConvert:
    """Test the convert method of OperatorAdapter."""

    @pytest.fixture
    def adapter(self):
        """Fixture providing a clean adapter instance."""
        return OperatorAdapter()

    def test_convert_click_action(self, adapter):
        """Test converting click action."""
        data = {"type": "click", "x": 100, "y": 200, "button": "left"}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 100
        assert result.point.y == 200
        assert result.button == "left"

    def test_convert_click_action_default_values(self, adapter):
        """Test converting click action with default values."""
        data = {"type": "click"}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 0
        assert result.point.y == 0
        assert result.button == "left"

    def test_convert_click_action_mapped_button(self, adapter):
        """Test converting click action with mapped button."""
        data = {"type": "click", "x": 100, "y": 200, "button": "wheel"}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.button == "middle"

    def test_convert_double_click_action(self, adapter):
        """Test converting double click action."""
        data = {"type": "double_click", "x": 150, "y": 250}
        result = adapter.convert(data)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 150
        assert result.point.y == 250
        assert result.button == "left"
        assert result.pattern == [100]  # Double click pattern

    def test_convert_scroll_action(self, adapter):
        """Test converting scroll action."""
        data = {"type": "scroll", "x": 300, "y": 400, "scroll_x": 10, "scroll_y": -20}
        result = adapter.convert(data)

        assert isinstance(result, ScrollAction)
        assert result.point is not None
        assert result.scroll is not None
        assert result.point.x == 300
        assert result.point.y == 400
        assert result.scroll.x == 10
        assert result.scroll.y == -20

    def test_convert_scroll_action_default_values(self, adapter):
        """Test converting scroll action with default values."""
        data = {"type": "scroll"}
        result = adapter.convert(data)

        assert isinstance(result, ScrollAction)
        assert result.point is not None
        assert result.scroll is not None
        assert result.point.x == 0
        assert result.point.y == 0
        assert result.scroll.x == 0
        assert result.scroll.y == 0

    def test_convert_type_action(self, adapter):
        """Test converting type action."""
        data = {"type": "type", "text": "Hello, World!"}
        result = adapter.convert(data)

        assert isinstance(result, TypeAction)
        assert result.text == "Hello, World!"
        assert result.enter_after is False

    def test_convert_type_action_default_text(self, adapter):
        """Test converting type action with default text."""
        data = {"type": "type"}
        result = adapter.convert(data)

        assert isinstance(result, TypeAction)
        assert result.text == ""
        assert result.enter_after is False

    def test_convert_wait_action(self, adapter):
        """Test converting wait action."""
        data = {"type": "wait", "ms": 2000}
        result = adapter.convert(data)

        assert isinstance(result, WaitAction)
        assert result.time == 2000

    def test_convert_wait_action_default_time(self, adapter):
        """Test converting wait action with default time."""
        data = {"type": "wait"}
        result = adapter.convert(data)

        assert isinstance(result, WaitAction)
        assert result.time == 1000

    def test_convert_move_action(self, adapter):
        """Test converting move action."""
        data = {"type": "move", "x": 500, "y": 600}
        result = adapter.convert(data)

        assert isinstance(result, MoveAction)
        assert result.point is not None
        assert result.point.x == 500
        assert result.point.y == 600

    def test_convert_move_action_default_values(self, adapter):
        """Test converting move action with default values."""
        data = {"type": "move"}
        result = adapter.convert(data)

        assert isinstance(result, MoveAction)
        assert result.point is not None
        assert result.point.x == 0
        assert result.point.y == 0

    def test_convert_keypress_action(self, adapter):
        """Test converting keypress action."""
        data = {"type": "keypress", "keys": ["ctrl", "c"]}
        result = adapter.convert(data)

        assert isinstance(result, PressAction)
        assert result.keys == ["ctrl", "c"]

    def test_convert_keypress_action_mapped_keys(self, adapter):
        """Test converting keypress action with mapped keys."""
        data = {"type": "keypress", "keys": ["return", "arrowup"]}
        result = adapter.convert(data)

        assert isinstance(result, PressAction)
        assert result.keys == ["enter", "up"]

    def test_convert_keypress_action_default_keys(self, adapter):
        """Test converting keypress action with default keys."""
        data = {"type": "keypress"}
        result = adapter.convert(data)

        assert isinstance(result, PressAction)
        assert result.keys == []

    def test_convert_drag_action(self, adapter):
        """Test converting drag action."""
        data = {
            "type": "drag",
            "path": [{"x": 100, "y": 200}, {"x": 150, "y": 250}, {"x": 200, "y": 300}],
        }
        result = adapter.convert(data)

        assert isinstance(result, DragAction)
        assert len(result.path) == 3
        assert result.path[0].x == 100
        assert result.path[0].y == 200
        assert result.path[1].x == 150
        assert result.path[1].y == 250
        assert result.path[2].x == 200
        assert result.path[2].y == 300

    def test_convert_drag_action_default_path(self, adapter):
        """Test converting drag action with default path."""
        data = {"type": "drag"}
        result = adapter.convert(data)

        assert isinstance(result, DragAction)
        assert result.path == []

    def test_convert_drag_action_path_with_missing_coords(self, adapter):
        """Test converting drag action with missing coordinates."""
        data = {
            "type": "drag",
            "path": [
                {"x": 100},  # Missing y
                {"y": 200},  # Missing x
                {},  # Missing both
            ],
        }
        result = adapter.convert(data)

        assert isinstance(result, DragAction)
        assert len(result.path) == 3
        assert result.path[0].x == 100
        assert result.path[0].y == 0  # Default value
        assert result.path[1].x == 0  # Default value
        assert result.path[1].y == 200
        assert result.path[2].x == 0  # Default value
        assert result.path[2].y == 0  # Default value

    def test_convert_screenshot_action(self, adapter):
        """Test converting screenshot action."""
        data = {"type": "screenshot"}
        result = adapter.convert(data)

        assert isinstance(result, ScreenshotFetch)

    def test_convert_response_action(self, adapter):
        """Test converting response action."""
        data = {"type": "response", "text": "Task completed successfully"}
        result = adapter.convert(data)

        assert isinstance(result, ResponseAction)
        assert result.text == "Task completed successfully"

    def test_convert_response_action_default_text(self, adapter):
        """Test converting response action with default text."""
        data = {"type": "response"}
        result = adapter.convert(data)

        assert isinstance(result, ResponseAction)
        assert result.text == ""

    def test_convert_unsupported_action_type(self, adapter):
        """Test converting unsupported action type."""
        data = {"type": "unsupported_action"}

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Unsupported action type: unsupported_action" in str(exc_info.value)

    def test_convert_invalid_data_structure(self, adapter):
        """Test converting invalid data structure."""
        # Test with non-dict data
        with pytest.raises(ValueError) as exc_info:
            adapter.convert("invalid_data")

        assert "Invalid action" in str(exc_info.value)

    def test_convert_missing_type_field(self, adapter):
        """Test converting data without type field."""
        data = {"x": 100, "y": 200}  # Missing type

        with pytest.raises(ValueError) as exc_info:
            adapter.convert(data)

        assert "Unsupported action type: None" in str(exc_info.value)

    def test_convert_none_data(self, adapter):
        """Test converting None data."""
        with pytest.raises(ValueError) as exc_info:
            adapter.convert(None)

        assert "Invalid action" in str(exc_info.value)


class TestOperatorAdapterIntegration:
    """Integration tests for OperatorAdapter."""

    @pytest.fixture
    def adapter(self):
        """Fixture providing a clean adapter instance."""
        return OperatorAdapter()

    def test_full_click_pipeline(self, adapter):
        """Test full click action processing pipeline."""
        # Set adapter dimensions to avoid scaling
        adapter.agent_width = 1920
        adapter.agent_height = 1080
        adapter.env_width = 1920
        adapter.env_height = 1080

        # Test the full adapt method
        raw_action = {"type": "click", "x": 100, "y": 200, "button": "right"}

        result = adapter.adapt(raw_action)

        assert isinstance(result, ClickAction)
        assert result.point is not None
        assert result.point.x == 100
        assert result.point.y == 200
        assert result.button == "right"

        # Check that it was added to memory
        assert len(adapter.memory) == 1
        assert adapter.memory[0] == result

    def test_multiple_actions_processing(self, adapter):
        """Test processing multiple actions."""
        # Set adapter dimensions to avoid scaling
        adapter.agent_width = 1920
        adapter.agent_height = 1080
        adapter.env_width = 1920
        adapter.env_height = 1080

        actions = [
            {"type": "click", "x": 100, "y": 200},
            {"type": "type", "text": "hello"},
            {"type": "keypress", "keys": ["return"]},
        ]

        results = adapter.adapt_list(actions)

        assert len(results) == 3
        assert isinstance(results[0], ClickAction)
        assert isinstance(results[1], TypeAction)
        assert isinstance(results[2], PressAction)

        # Check memory
        assert len(adapter.memory) == 3
