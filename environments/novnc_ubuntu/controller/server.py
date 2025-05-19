from io import BytesIO
import pyautogui
from controller.pyautogui_rosetta import PyAutoGUIRosetta

from typing import Any
from mcp.server.fastmcp import FastMCP, Image

server = FastMCP("StatefulServer", port=8483)


@server.tool("step")
def step(actions: list[dict[str, Any]]) -> Image:
    pyautogui_rosetta = PyAutoGUIRosetta()
    pyautogui_rosetta.execute_sequence(actions)

    buffer = BytesIO()
    pyautogui.screenshot().save(buffer, format="PNG")
    return Image(data=buffer.getvalue(), format="png")


if __name__ == "__main__":
    server.run(transport="streamable-http")
