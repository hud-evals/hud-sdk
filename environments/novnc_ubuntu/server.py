import base64
from io import BytesIO
import pyautogui
from .pyautogui_rosetta import PyAutoGUIRosetta

from typing import Any
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("StatefulServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

def screenshot_base64() -> str:
    """
    Take a screenshot and return it as a base64 encoded string.
    """
    photo = pyautogui.screenshot()
    output = BytesIO()
    photo.save(output, format="PNG")
    im_data = output.getvalue()

    image_data = base64.b64encode(im_data).decode()
    return image_data

@mcp.tool("step")
def step(action: list[dict[str, Any]]) -> Any:
    """
    Execute a sequence of actions.
    """
    pyautogui_rosetta = PyAutoGUIRosetta()
    pyautogui_rosetta.execute_sequence(action)

    screenshot = screenshot_base64()

    return {"observation": {"screenshot": screenshot}}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
