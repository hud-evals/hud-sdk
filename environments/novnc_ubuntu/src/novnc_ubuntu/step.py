from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import pyautogui

from .pyautogui_rosetta import PyAutoGUIRosetta


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


def step(action: list[dict[str, Any]]) -> None:
    """
    Execute a sequence of actions.
    """
    pyautogui_rosetta = PyAutoGUIRosetta()
    pyautogui_rosetta.execute_sequence(action)

    screenshot = screenshot_base64()

    return {"observation": {"screenshot": screenshot}}

