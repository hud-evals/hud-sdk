from __future__ import annotations

from io import BytesIO
from typing import Any

import pyautogui
from mcp.server.fastmcp import FastMCP, Image

from controller.pyautogui_rosetta import PyAutoGUIRosetta

server = FastMCP("StatefulServer", port=8483)


@server.tool("step")
def step(action: dict[str, Any]) -> Image:
    pyautogui_rosetta = PyAutoGUIRosetta()
    pyautogui_rosetta.execute_sequence([action])

    buffer = BytesIO()
    pyautogui.screenshot().save(buffer, format="PNG")
    return Image(data=buffer.getvalue(), format="png")


if __name__ == "__main__":
    server.run(transport="streamable-http")
