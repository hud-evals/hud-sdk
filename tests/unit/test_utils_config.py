import pytest

from hud.utils.config import expand_config
from hud.utils.common import FunctionConfig

@pytest.mark.parametrize(
    "raw, expected_funcs",
    [
        ("browser.refresh", ["browser.refresh"]),
    ],
)
def test_expand_config_permutations(raw, expected_funcs):
    """expand_config should normalise any accepted format to FunctionConfig list."""

    out = expand_config(raw)
    assert [cfg.function for cfg in out] == expected_funcs
    # Ensure every element is a FunctionConfig
    assert all(isinstance(cfg, FunctionConfig) for cfg in out)

def test_expand_config_string():
    """Test string format (function name only)."""
    raw = "browser.refresh"
    out = expand_config(raw)
    assert len(out) == 1
    assert isinstance(out[0], FunctionConfig)
    assert out[0].function == "browser.refresh"
    assert out[0].args == []

def test_expand_config_tuple():
    """Test tuple format (function name and args)."""
    raw = ("goto", "example.com", True)
    out = expand_config(raw)
    assert len(out) == 1
    assert isinstance(out[0], FunctionConfig)
    assert out[0].function == "goto"
    assert out[0].args == ["example.com", True]

def test_expand_config_dict():
    """Test dictionary format."""
    raw = {"function": "click", "args": ["#btn"]}
    out = expand_config(raw)
    assert len(out) == 1
    assert isinstance(out[0], FunctionConfig)
    assert out[0].function == "click"
    assert out[0].args == ["#btn"]

def test_expand_config_dict_single_arg():
    """Test dictionary format with non-list args."""
    raw = {"function": "click", "args": "#btn"}
    out = expand_config(raw)
    assert len(out) == 1
    assert isinstance(out[0], FunctionConfig)
    assert out[0].function == "click"
    assert out[0].args == ["#btn"]

def test_expand_config_function_config():
    """Test passing a FunctionConfig directly."""
    raw = FunctionConfig(function="test", args=["arg1"])
    out = expand_config(raw)
    assert len(out) == 1
    assert isinstance(out[0], FunctionConfig)
    assert out[0].function == "test"
    assert out[0].args == ["arg1"]

def test_expand_config_function_config_list():
    """Test list of FunctionConfig objects."""
    raw = [
        FunctionConfig(function="test1", args=[]),
        FunctionConfig(function="test2", args=["arg"])
    ]
    out = expand_config(raw)
    assert len(out) == 2
    assert all(isinstance(cfg, FunctionConfig) for cfg in out)
    assert [cfg.function for cfg in out] == ["test1", "test2"]
    assert out[1].args == ["arg"]

def test_expand_config_invalid_list():
    """Test that mixed lists are not supported."""
    raw = ["chrome.maximize", ("goto", "page")]
    with pytest.raises(ValueError):
        expand_config(raw)

def test_expand_config_invalid_tuple():
    """Test tuple must start with string."""
    raw = (123, "arg")
    with pytest.raises(ValueError):
        expand_config(raw)

def test_expand_config_invalid_dict():
    """Test dict must have function key as string."""
    raw = {"function": 123, "args": []}
    with pytest.raises(ValueError):
        expand_config(raw)
