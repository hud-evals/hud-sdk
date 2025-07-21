from __future__ import annotations

import pytest
from mcp.types import ImageContent, TextContent

from hud.tools.computer.hud import HudComputerTool

# (action, kwargs)
CASES = [
    ("screenshot", {}),
    ("click", {"x": 1, "y": 1, "pattern": []}),
    ("press", {"keys": ["ctrl", "c"]}),
    ("keydown", {"keys": ["shift"]}),
    ("keyup", {"keys": ["shift"]}),
    ("type", {"text": "hello"}),
    ("scroll", {"scroll_y": 20}),
    ("move", {"x": 5, "y": 5}),
    ("wait", {"time": 5}),
    ("drag", {"path": [(0, 0), (10, 10)]}),
    ("mouse_down", {}),
    ("mouse_up", {}),
    ("hold_key", {"text": "a", "duration": 0.1}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("action, params", CASES)
async def test_hud_computer_actions(action: str, params: dict):
    comp = HudComputerTool()
    blocks = await comp(action=action, **params)
    # Ensure at least one content block is returned
    assert blocks
    assert all(isinstance(b, ImageContent | TextContent) for b in blocks)
