from __future__ import annotations

import pytest

from hud.utils.common import FunctionConfig
from hud.utils.config import expand_config


@pytest.mark.parametrize(
    "config, expected",
    [
        ("test", [{"function": "test", "args": [], "id": None}]),
        (("test",), [{"function": "test", "args": [], "id": None}]),
        (
            [FunctionConfig(function="test", args=[])],
            [{"function": "test", "args": [], "id": None}],
        ),
        ({"function": "test", "args": []}, [{"function": "test", "args": [], "id": None}]),
        (
            {"function": "test", "args": ["arg1"]},
            [{"function": "test", "args": ["arg1"], "id": None}],
        ),
        (
            {"function": "test", "args": ["arg1"], "id": "test_id"},
            [{"function": "test", "args": ["arg1"], "id": "test_id"}],
        ),
        (("test", "arg1", "arg2"), [{"function": "test", "args": ["arg1", "arg2"], "id": None}]),
    ],
)
def test_expand_config(config, expected):
    result = expand_config(config)
    assert len(result) == len(expected)
    for i, item in enumerate(result):
        assert item.function == expected[i]["function"]
        assert item.args == expected[i]["args"]
        assert item.id == expected[i]["id"]
