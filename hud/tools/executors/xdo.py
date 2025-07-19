import asyncio
import base64
import os
import shlex
from pathlib import Path
from uuid import uuid4

from ..base import ToolResult
from ..utils import run

OUTPUT_DIR = os.environ.get("SCREENSHOT_DIR", "/tmp/outputs")


class XDOExecutor:
    """
    Low-level executor for xdotool commands.
    Handles display management and screenshot capture.
    """
    
    def __init__(self, display_num: int | None = None):
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
    
    async def execute(self, command: str, take_screenshot: bool = True) -> ToolResult:
        """
        Execute an xdotool command.
        
        Args:
            command: The xdotool command (without xdotool prefix)
            take_screenshot: Whether to capture a screenshot after execution
        
        Returns:
            ToolResult with output, error, and optional screenshot
        """
        full_command = f"{self.xdotool} {command}"
        
        # Execute command
        returncode, stdout, stderr = await run(full_command)
        
        # Prepare result
        result = ToolResult(
            output=stdout if stdout else None,
            error=stderr if stderr or returncode != 0 else None
        )
        
        # Take screenshot if requested
        if take_screenshot:
            await asyncio.sleep(self._screenshot_delay)
            screenshot = await self.screenshot()
            if screenshot:
                result = ToolResult(
                    output=result.output,
                    error=result.error,
                    base64_image=screenshot
                )
        
        return result
    
    async def screenshot(self) -> str | None:
        """
        Take a screenshot and return base64 encoded image.
        
        Returns:
            Base64 encoded PNG image or None if failed
        """
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        screenshot_path = output_dir / f"screenshot_{uuid4().hex}.png"
        screenshot_cmd = f"{self._display_prefix}scrot -p {screenshot_path}"
        
        returncode, _, stderr = await run(screenshot_cmd)
        
        if returncode == 0 and screenshot_path.exists():
            try:
                image_data = screenshot_path.read_bytes()
                screenshot_path.unlink()  # Clean up
                return base64.b64encode(image_data).decode()
            except Exception:
                return None
        
        return None
    
    async def mouse_move(self, x: int, y: int, take_screenshot: bool = True) -> ToolResult:
        """Move mouse to specified coordinates."""
        return await self.execute(f"mousemove {x} {y}", take_screenshot=take_screenshot)
    
    async def click(self, button: int = 1, x: int | None = None, y: int | None = None, take_screenshot: bool = True) -> ToolResult:
        """Click at specified coordinates or current position."""
        if x is not None and y is not None:
            return await self.execute(f"mousemove {x} {y} click {button}", take_screenshot=take_screenshot)
        else:
            return await self.execute(f"click {button}", take_screenshot=take_screenshot)
    
    async def type_text(self, text: str, delay: int = 12, take_screenshot: bool = True) -> ToolResult:
        """Type text with specified delay between keystrokes."""
        # Escape text for shell
        escaped_text = shlex.quote(text)
        return await self.execute(f"type --delay {delay} -- {escaped_text}", take_screenshot=take_screenshot)
    
    async def key(self, key_sequence: str, take_screenshot: bool = True) -> ToolResult:
        """Press a key or key combination."""
        return await self.execute(f"key -- {key_sequence}", take_screenshot=take_screenshot)
    
    async def scroll(self, direction: str, amount: int = 5, x: int | None = None, y: int | None = None, take_screenshot: bool = True) -> ToolResult:
        """
        Scroll in specified direction.
        
        Args:
            direction: "up", "down", "left", or "right"
            amount: Number of scroll clicks
            x, y: Optional coordinates to scroll at
            take_screenshot: Whether to capture a screenshot after execution
        """
        scroll_button_map = {
            "up": 4,
            "down": 5,
            "left": 6,
            "right": 7
        }
        
        button = scroll_button_map.get(direction, 5)  # Default to down
        
        if x is not None and y is not None:
            return await self.execute(f"mousemove {x} {y} click --repeat {amount} {button}", take_screenshot=take_screenshot)
        else:
            return await self.execute(f"click --repeat {amount} {button}", take_screenshot=take_screenshot)
    
    async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, take_screenshot: bool = True) -> ToolResult:
        """Drag from start coordinates to end coordinates."""
        return await self.execute(
            f"mousemove {start_x} {start_y} mousedown 1 mousemove {end_x} {end_y} mouseup 1",
            take_screenshot=take_screenshot
        ) 