from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from hud.adapters.common import Adapter
from hud.adapters.common.types import ClickAction, Point, TypeAction


@pytest.fixture
def adapter():
    """Fixture providing a clean adapter instance."""
    return Adapter()


@pytest.fixture
def test_image():
    """Fixture providing test image in various formats."""
    img = Image.new("RGB", (100, 80), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    img_array = np.array(img)

    return {
        "pil": img,
        "bytes": img_bytes.getvalue(),
        "base64": img_base64,
        "array": img_array,
    }


def test_init(adapter):
    """Test adapter initialization."""
    assert adapter.agent_width == 1920
    assert adapter.agent_height == 1080
    assert adapter.env_width == 1920
    assert adapter.env_height == 1080
    assert adapter.memory == []


def test_preprocess(adapter):
    """Test preprocess method (default implementation)."""
    action = {"type": "click", "point": {"x": 100, "y": 100}}
    result = adapter.preprocess(action)
    assert result == action  # Default implementation returns unchanged


def test_convert_valid(adapter):
    """Test convert method with valid action."""
    action = ClickAction(point=Point(x=100, y=100))
    result = adapter.convert(action)
    # Fix: Instead of checking against CLA, check it's the same type as the input
    assert isinstance(result, ClickAction)
    assert result == action


def test_convert_invalid(adapter):
    """Test convert method with invalid action."""
    with pytest.raises(ValueError):
        adapter.convert(None)  # type: ignore


def test_json_valid(adapter):
    """Test json method with valid action."""
    action = ClickAction(point=Point(x=100, y=100))
    result = adapter.json(action)
    assert isinstance(result, dict)
    assert result["type"] == "click"
    assert result["point"]["x"] == 100
    assert result["point"]["y"] == 100


def test_json_invalid(adapter):
    """Test json method with invalid action."""
    with pytest.raises(ValueError):
        adapter.json(None)  # type: ignore


def test_rescale_pil_image(adapter, test_image):
    """Test rescaling PIL Image."""
    result = adapter.rescale(test_image["pil"])

    # Verify result is base64 string
    assert isinstance(result, str)

    # Verify environment dimensions were updated
    assert adapter.env_width == 100
    assert adapter.env_height == 80

    # Decode and verify image dimensions
    img_bytes = base64.b64decode(result)
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (adapter.agent_width, adapter.agent_height)


def test_rescale_numpy_array(adapter, test_image):
    """Test rescaling numpy array."""
    result = adapter.rescale(test_image["array"])

    # Verify result is base64 string
    assert isinstance(result, str)

    # Verify environment dimensions were updated
    assert adapter.env_width == 100
    assert adapter.env_height == 80


def test_rescale_base64(adapter, test_image):
    """Test rescaling base64 string."""
    result = adapter.rescale(test_image["base64"])

    # Verify result is base64 string
    assert isinstance(result, str)

    # Verify environment dimensions were updated
    assert adapter.env_width == 100
    assert adapter.env_height == 80


def test_rescale_base64_with_header(adapter, test_image):
    """Test rescaling base64 string with header."""
    base64_with_header = f"data:image/png;base64,{test_image['base64']}"
    result = adapter.rescale(base64_with_header)

    # Verify result is base64 string
    assert isinstance(result, str)

    # Verify environment dimensions were updated
    assert adapter.env_width == 100
    assert adapter.env_height == 80


def test_rescale_invalid_type(adapter):
    """Test rescaling with invalid type."""
    with pytest.raises(ValueError):
        adapter.rescale(123)  # type: ignore


def test_rescale_none(adapter):
    """Test rescaling with None."""
    result = adapter.rescale(None)
    assert result is None


def test_postprocess_action_click(adapter):
    """Test postprocess_action with click action."""
    # Set different agent and env dimensions
    adapter.agent_width = 1000
    adapter.agent_height = 800
    adapter.env_width = 2000
    adapter.env_height = 1600

    action = {"type": "click", "point": {"x": 500, "y": 400}}
    result = adapter.postprocess_action(action)

    # Coordinates should be doubled
    assert result["point"]["x"] == 1000
    assert result["point"]["y"] == 800


def test_postprocess_action_drag(adapter):
    """Test postprocess_action with drag action."""
    # Set different agent and env dimensions
    adapter.agent_width = 1000
    adapter.agent_height = 800
    adapter.env_width = 2000
    adapter.env_height = 1600

    action = {"type": "drag", "path": [{"x": 100, "y": 200}, {"x": 300, "y": 400}]}
    result = adapter.postprocess_action(action)

    # Coordinates should be doubled
    assert result["path"][0]["x"] == 200
    assert result["path"][0]["y"] == 400
    assert result["path"][1]["x"] == 600
    assert result["path"][1]["y"] == 800


def test_postprocess_action_scroll(adapter):
    """Test postprocess_action with scroll action."""
    # Set different agent and env dimensions
    adapter.agent_width = 1000
    adapter.agent_height = 800
    adapter.env_width = 2000
    adapter.env_height = 1600

    action = {"type": "scroll", "point": {"x": 500, "y": 400}, "scroll": {"x": 0, "y": 10}}
    result = adapter.postprocess_action(action)

    # Point coordinates should be doubled
    assert result["point"]["x"] == 1000
    assert result["point"]["y"] == 800
    # Scroll amount should be scaled
    assert result["scroll"]["x"] == 0
    assert result["scroll"]["y"] == 20


def test_postprocess_action_empty(adapter):
    """Test postprocess_action with empty action."""
    result = adapter.postprocess_action({})
    assert result == {}


def test_adapt(adapter):
    """Test adapt method."""
    # Mock the needed methods
    with (
        patch.object(adapter, "preprocess", return_value={"preprocessed": True}),
        patch.object(adapter, "convert", return_value=TypeAction(text="test")),
        patch.object(adapter, "json", return_value={"type": "type", "text": "test"}),
        patch.object(adapter, "postprocess_action", return_value={"type": "type", "text": "test"}),
        patch("hud.adapters.common.adapter.TypeAdapter") as mock_adapter,
    ):
        mock_validator = MagicMock()
        mock_adapter.return_value = mock_validator
        mock_validator.validate_python.return_value = TypeAction(text="test")

        adapter.adapt({"raw": "action"})

        # Verify the method chain was called correctly
        adapter.preprocess.assert_called_once_with({"raw": "action"})
        adapter.convert.assert_called_once_with({"preprocessed": True})
        adapter.json.assert_called_once_with(TypeAction(text="test"))
        adapter.postprocess_action.assert_called_once_with({"type": "type", "text": "test"})

        # Verify the memory was updated
        assert len(adapter.memory) == 1
        assert adapter.memory[0] == TypeAction(text="test")


def test_adapt_list(adapter):
    """Test adapt_list method."""
    # Fix: Use side_effect to return different values for each call to adapt
    click_action = ClickAction(point=Point(x=100, y=100))
    type_action = TypeAction(text="test")

    mock_adapt = MagicMock(side_effect=[click_action, type_action])
    with patch.object(adapter, "adapt", mock_adapt):
        actions = [{"type": "click"}, {"type": "type"}]
        result = adapter.adapt_list(actions, None)

        assert adapter.adapt.call_count == 2
        assert len(result) == 2
        assert result[0] == click_action
        assert result[1] == type_action


def test_adapt_list_invalid(adapter):
    """Test adapt_list with invalid input."""
    with pytest.raises(ValueError):
        adapter.adapt_list("not a list", None)  # type: ignore


def test_integration(adapter):
    """Integration test for the full adapter pipeline."""
    adapter.agent_width = 1000
    adapter.agent_height = 800
    adapter.env_width = 2000
    adapter.env_height = 1600

    # Create a click action
    action = ClickAction(point=Point(x=500, y=400))

    result = adapter.adapt(action)

    assert isinstance(result, ClickAction)
    assert result.point is not None
    assert result.point.x == 1000  # Scaled from 500 to 1000
    assert result.point.y == 800  # Scaled from 400 to 800

    assert len(adapter.memory) == 1
