from __future__ import annotations

import asyncio
import base64
import logging
import os
import shlex
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from hud.tools.base import ToolResult
from hud.tools.utils import run

from .base import BaseExecutor

OUTPUT_DIR = os.environ.get("SCREENSHOT_DIR")
logger = logging.getLogger(__name__)


class XDOExecutor(BaseExecutor):
    """
    Low-level executor for xdotool commands.
    Handles display management and screenshot capture.
    Falls back to simulation mode when no display is available.
    """

    def __init__(self, display_num: int | None = None) -> None:
        """
        Initialize the executor.

        Args:
            display_num: X display number (e.g. 0 for :0)
        """
        self.display_num = display_num
        if display_num is not None:
            self._display_prefix = f"DISPLAY=:{display_num} "
        else:
            self._display_prefix = ""

        self.xdotool = f"{self._display_prefix}xdotool"
        self._screenshot_delay = 0.5  # Delay before taking screenshots

        # Check if display is available
        self.is_simulation = not self._check_display_available()

        if self.is_simulation:
            logger.warning("No X11 display available - running in simulation mode")
            # Initialize parent BaseExecutor
            super().__init__(display_num)
        else:
            logger.info("X11 display available - running in real mode")

    def _check_display_available(self) -> bool:
        """
        Check if X11 display is available.

        Returns:
            True if display is available, False otherwise
        """
        display = os.environ.get("DISPLAY")
        if not display:
            return False

        # Try a simple xdotool command to test display availability
        try:
            import subprocess

            result = subprocess.run(  # noqa: S603
                [self.xdotool.split()[-1], "getdisplaygeometry"],
                capture_output=True,
                timeout=2,
                env={**os.environ, **({"DISPLAY": display} if not self._display_prefix else {})},
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    async def execute(self, command: str, take_screenshot: bool = True) -> ToolResult:
        """
        Execute an xdotool command or simulate it if no display available.

        Args:
            command: The xdotool command (without xdotool prefix)
            take_screenshot: Whether to capture a screenshot after execution

        Returns:
            ToolResult with output, error, and optional screenshot
        """
        # Fall back to simulation if no display
        if self.is_simulation:
            return await super().execute(command, take_screenshot)

        # Real execution
        full_command = f"{self.xdotool} {command}"

        # Execute command
        returncode, stdout, stderr = await run(full_command)

        # Prepare result
        result = ToolResult(
            output=stdout if stdout else None, error=stderr if stderr or returncode != 0 else None
        )

        # Take screenshot if requested
        if take_screenshot:
            await asyncio.sleep(self._screenshot_delay)
            screenshot = await self.screenshot()
            if screenshot:
                result = ToolResult(
                    output=result.output, error=result.error, base64_image=screenshot
                )

        return result

    async def screenshot(self) -> str | None:
        """
        Take a screenshot and return base64 encoded image.

        Returns:
            Base64 encoded PNG image or None if failed
        """
        # Fall back to simulation if no display
        if self.is_simulation:
            return await super().screenshot()

        # Real screenshot
        if OUTPUT_DIR:
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = output_dir / f"screenshot_{uuid4().hex}.png"
        else:
            # Generate a unique path in system temp dir without opening a file
            screenshot_path = Path(gettempdir()) / f"screenshot_{uuid4().hex}.png"

        screenshot_cmd = f"{self._display_prefix}scrot -p {screenshot_path}"

        returncode, _, stderr = await run(screenshot_cmd)

        if returncode == 0 and screenshot_path.exists():
            try:
                image_data = screenshot_path.read_bytes()
                # Remove the file unless user requested persistence via env var
                if not OUTPUT_DIR:
                    screenshot_path.unlink(missing_ok=True)
                return base64.b64encode(image_data).decode()
            except Exception:
                return None

        return None

    async def mouse_move(self, x: int, y: int, take_screenshot: bool = True) -> ToolResult:
        """Move mouse to specified coordinates."""
        return await self.execute(f"mousemove {x} {y}", take_screenshot=take_screenshot)

    async def click(
        self,
        button: int = 1,
        x: int | None = None,
        y: int | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Click at specified coordinates or current position."""
        if x is not None and y is not None:
            return await self.execute(
                f"mousemove {x} {y} click {button}", take_screenshot=take_screenshot
            )
        else:
            return await self.execute(f"click {button}", take_screenshot=take_screenshot)

    async def type_text(
        self, text: str, delay: int = 12, take_screenshot: bool = True
    ) -> ToolResult:
        """Type text with specified delay between keystrokes."""
        # Escape text for shell
        escaped_text = shlex.quote(text)
        return await self.execute(
            f"type --delay {delay} -- {escaped_text}", take_screenshot=take_screenshot
        )

    async def key(self, key_sequence: str, take_screenshot: bool = True) -> ToolResult:
        """Press a key or key combination."""
        return await self.execute(f"key -- {key_sequence}", take_screenshot=take_screenshot)

    async def scroll(
        self,
        direction: str,
        amount: int = 5,
        x: int | None = None,
        y: int | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """
        Scroll in specified direction.

        Args:
            direction: "up", "down", "left", or "right"
            amount: Number of scroll clicks
            x, y: Optional coordinates to scroll at
            take_screenshot: Whether to capture a screenshot after execution
        """
        scroll_button_map = {"up": 4, "down": 5, "left": 6, "right": 7}

        button = scroll_button_map.get(direction, 5)  # Default to down

        if x is not None and y is not None:
            return await self.execute(
                f"mousemove {x} {y} click --repeat {amount} {button}",
                take_screenshot=take_screenshot,
            )
        else:
            return await self.execute(
                f"click --repeat {amount} {button}", take_screenshot=take_screenshot
            )

    async def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, take_screenshot: bool = True
    ) -> ToolResult:
        """Drag from start coordinates to end coordinates."""
        return await self.execute(
            f"mousemove {start_x} {start_y} mousedown 1 mousemove {end_x} {end_y} mouseup 1",
            take_screenshot=take_screenshot,
        )
