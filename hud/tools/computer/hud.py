# flake8: noqa: B008
from __future__ import annotations

import asyncio
import logging
from typing import Literal

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ImageContent, TextContent
from pydantic import Field

from hud.tools.base import ToolError, ToolResult, tool_result_to_content_blocks
from hud.tools.executors.xdo import XDOExecutor

logger = logging.getLogger(__name__)


class HudComputerTool:
    """
    A tool that provides computer control methods matching CLA actions.
    Each method corresponds to a specific CLA action type.
    """

    def __init__(
        self, width: int = 1920, height: int = 1080, display_num: int | None = None
    ) -> None:
        """
        Initialize the HUD computer tool.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            display_num: X display number
        """
        self.width = width
        self.height = height
        self.executor = XDOExecutor(display_num=display_num)

    async def __call__(
        self,
        action: str = Field(..., description="The action name (click, type, move, etc.)"),
        # Click parameters
        x: int | None = Field(None, description="X coordinate for click/move/scroll actions"),
        y: int | None = Field(None, description="Y coordinate for click/move/scroll actions"),
        button: Literal["left", "right", "middle", "back", "forward"] | None = Field(
            None, description="Mouse button for click actions"
        ),
        pattern: list[int] | None = Field(
            None, description="Click pattern for multi-clicks (e.g., [100] for double-click)"
        ),
        # Key/Type parameters
        text: str | None = Field(None, description="Text for type/response actions"),
        keys: list[str] | None = Field(None, description="Keys for press/keydown/keyup actions"),
        enter_after: bool | None = Field(None, description="Whether to press Enter after typing"),
        # Scroll parameters
        scroll_x: int | None = Field(
            None, description="Horizontal scroll amount (positive = right)"
        ),
        scroll_y: int | None = Field(None, description="Vertical scroll amount (positive = down)"),
        # Move parameters
        offset_x: int | None = Field(None, description="X offset for relative move"),
        offset_y: int | None = Field(None, description="Y offset for relative move"),
        # Drag parameters
        path: list[tuple[int, int]] | None = Field(
            None, description="Path for drag actions as list of (x, y) coordinates"
        ),
        # Wait parameter
        time: int | None = Field(None, description="Time in milliseconds for wait action"),
        # General parameters
        hold_keys: list[str] | None = Field(None, description="Keys to hold during action"),
        # hold_key specific
        duration: float | None = Field(None, description="Duration in seconds for hold_key action"),
    ) -> list[ImageContent | TextContent]:
        """
        Execute a computer control action by name.

        Returns:
            List of MCP content blocks
        """
        logger.info("HudComputerTool executing action: %s", action)

        try:
            # Execute the action directly
            if action == "click":
                result = await self.click(
                    x=x, y=y, button=button or "left", pattern=pattern, hold_keys=hold_keys
                )

            elif action == "press":
                if keys is None:
                    raise ToolError("keys parameter is required for press")
                result = await self.press(keys=keys)

            elif action == "keydown":
                if keys is None:
                    raise ToolError("keys parameter is required for keydown")
                result = await self.keydown(keys=keys)

            elif action == "keyup":
                if keys is None:
                    raise ToolError("keys parameter is required for keyup")
                result = await self.keyup(keys=keys)

            elif action == "type":
                if text is None:
                    raise ToolError("text parameter is required for type")
                result = await self.type(text=text, enter_after=enter_after or False)

            elif action == "scroll":
                result = await self.scroll(
                    x=x, y=y, scroll_x=scroll_x, scroll_y=scroll_y, hold_keys=hold_keys
                )

            elif action == "move":
                result = await self.move(x=x, y=y, offset_x=offset_x, offset_y=offset_y)

            elif action == "wait":
                if time is None:
                    raise ToolError("time parameter is required for wait")
                result = await self.wait(time=time)

            elif action == "drag":
                if path is None:
                    raise ToolError("path parameter is required for drag")
                result = await self.drag(path=path, pattern=pattern, hold_keys=hold_keys)

            elif action == "response":
                if text is None:
                    raise ToolError("text parameter is required for response")
                # Response returns content blocks directly
                return await self.response(text=text)

            elif action == "screenshot":
                result = await self.screenshot()

            elif action == "position":
                result = await self.position()

            elif action == "hold_key":
                if text is None:
                    raise ToolError("text parameter is required for hold_key")
                if duration is None:
                    raise ToolError("duration parameter is required for hold_key")
                result = await self.hold_key(text=text, duration=duration)

            elif action == "mouse_down":
                result = await self.mouse_down(button=button or "left")

            elif action == "mouse_up":
                result = await self.mouse_up(button=button or "left")

            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown action: {action}"))

            # Handle screenshot capture based on action type
            # Actions that should always include a screenshot
            screenshot_actions = {
                "screenshot",
                "click",
                "type",
                "scroll",
                "move",
                "drag",
                "press",
                "keydown",
                "keyup",
                "wait",
                "hold_key",
                "mouse_down",
                "mouse_up",
            }

            if action in screenshot_actions and action != "screenshot":
                # Take a screenshot after the action
                screenshot_base64 = await self.executor.screenshot()
                if screenshot_base64 and isinstance(result, ToolResult):
                    # Add screenshot to the result
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot_base64
                    )

            # Convert result to content blocks
            if isinstance(result, ToolResult):
                return tool_result_to_content_blocks(result)
            else:
                # For actions that return content blocks directly (response)
                return result

        except TypeError as e:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message=f"Invalid parameters for {action}: {e!s}")
            ) from e

    # ClickAction
    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
    ) -> ToolResult:
        """
        Click at specified coordinates.

        Args:
            x, y: Coordinates to click at
            button: Mouse button to use
            pattern: List of delays for multi-clicks (e.g., [100] for double-click)
            hold_keys: Keys to hold during click
        """
        button_map = {"left": 1, "right": 3, "middle": 2, "back": 8, "forward": 9}
        button_num = button_map.get(button, 1)

        # Handle multi-clicks based on pattern
        if pattern:
            click_count = len(pattern) + 1
            delay = pattern[0] if pattern else 10
            if x is not None and y is not None:
                result = await self.executor.execute(
                    f"mousemove {x} {y} click --repeat {click_count} --delay {delay} {button_num}",
                    take_screenshot=False,
                )
            else:
                result = await self.executor.execute(
                    f"click --repeat {click_count} --delay {delay} {button_num}",
                    take_screenshot=False,
                )
        else:
            # Single click
            if x is not None and y is not None:
                result = await self.executor.execute(
                    f"mousemove {x} {y} click {button_num}", take_screenshot=False
                )
            else:
                result = await self.executor.execute(f"click {button_num}", take_screenshot=False)

        return result

    # PressAction
    async def press(self, keys: list[str]) -> ToolResult:
        """Press a key combination."""
        key_combo = "+".join(keys)
        result = await self.executor.key(key_combo, take_screenshot=False)
        return result

    # KeyDownAction
    async def keydown(self, keys: list[str]) -> ToolResult:
        """Press and hold keys."""
        last_result = None
        for key in keys:
            import shlex

            escaped_key = shlex.quote(key)
            last_result = await self.executor.execute(
                f"keydown {escaped_key}", take_screenshot=False
            )

        return last_result or ToolResult()

    # KeyUpAction
    async def keyup(self, keys: list[str]) -> ToolResult:
        """Release keys."""
        last_result = None
        for key in keys:
            import shlex

            escaped_key = shlex.quote(key)
            last_result = await self.executor.execute(f"keyup {escaped_key}", take_screenshot=False)

        return last_result or ToolResult()

    # TypeAction
    async def type(self, text: str, enter_after: bool = False) -> ToolResult:
        """Type text using the keyboard."""
        result = await self.executor.type_text(text, take_screenshot=False)
        if enter_after:
            # Type Enter after the text
            enter_result = await self.executor.key("Return", take_screenshot=False)
            # Combine outputs
            combined_output = (result.output or "") + "\n" + (enter_result.output or "")
            combined_error = None
            if result.error or enter_result.error:
                combined_error = (result.error or "") + "\n" + (enter_result.error or "")
            return ToolResult(output=combined_output.strip(), error=combined_error)
        return result

    # ScrollAction
    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
        hold_keys: list[str] | None = None,
    ) -> ToolResult:
        """
        Scroll at specified position.

        Args:
            x, y: Position to scroll at
            scroll_x, scroll_y: Scroll amounts (positive = right/down)
            hold_keys: Keys to hold during scroll
        """
        # Convert scroll amounts to direction and amount
        if scroll_y and scroll_y != 0:
            direction = "down" if scroll_y > 0 else "up"
            amount = abs(scroll_y)
            result = await self.executor.scroll(direction, amount, x, y, take_screenshot=False)
            return result
        elif scroll_x and scroll_x != 0:
            direction = "right" if scroll_x > 0 else "left"
            amount = abs(scroll_x)
            result = await self.executor.scroll(direction, amount, x, y, take_screenshot=False)
            return result
        else:
            # No scroll amount specified
            return ToolResult(output="No scroll amount specified")

    # MoveAction
    async def move(
        self,
        x: int | None = None,
        y: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
    ) -> ToolResult:
        """
        Move mouse cursor.

        Args:
            x, y: Absolute coordinates to move to
            offset_x, offset_y: Relative offset from current position
        """
        if x is not None and y is not None:
            # Absolute move
            result = await self.executor.mouse_move(x, y, take_screenshot=False)
        elif offset_x is not None or offset_y is not None:
            # Relative move - get current position first
            await self.executor.execute("getmouselocation", take_screenshot=False)
            # Parse current position (simplified - real impl would parse properly)
            # For now, just do an absolute move as a placeholder
            result = await self.executor.execute("getmouselocation", take_screenshot=False)
        else:
            result = ToolResult(output="No move coordinates specified")

        return result

    # WaitAction
    async def wait(self, time: int) -> ToolResult:
        """
        Wait for specified time.

        Args:
            time: Time to wait in milliseconds
        """
        duration_seconds = time / 1000.0
        await asyncio.sleep(duration_seconds)
        return ToolResult(output=f"Waited {time}ms")

    # DragAction
    async def drag(
        self,
        path: list[tuple[int, int]],
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
    ) -> ToolResult:
        """
        Drag along a path.

        Args:
            path: List of (x, y) coordinates defining the drag path
            pattern: Delays between path points
            hold_keys: Keys to hold during drag
        """
        if len(path) < 2:
            return ToolResult(error="Drag path must have at least 2 points")

        start_x, start_y = path[0]
        end_x, end_y = path[-1]

        # For now, simple drag from start to end
        # TODO: Handle intermediate points and patterns
        result = await self.executor.drag(start_x, start_y, end_x, end_y, take_screenshot=False)
        return result

    # ResponseAction (handled by agent, not computer tool)
    async def response(self, text: str) -> list[ImageContent | TextContent]:
        """Return a text response."""
        return [TextContent(text=text, type="text")]

    # ScreenshotFetch
    async def screenshot(self) -> ToolResult:
        """Take a screenshot of the current screen."""
        screenshot_base64 = await self.executor.screenshot()
        if screenshot_base64:
            return ToolResult(base64_image=screenshot_base64)
        else:
            return ToolResult(error="Failed to take screenshot")

    # PositionFetch
    async def position(self) -> ToolResult:
        """Get current cursor position."""
        result = await self.executor.execute("getmouselocation", take_screenshot=False)
        return result

    # mouse_down - generic mouse button down
    async def mouse_down(
        self, button: Literal["left", "right", "middle", "back", "forward"] = "left"
    ) -> ToolResult:
        """
        Press and hold a mouse button.

        Args:
            button: Mouse button to press
        """
        button_map = {"left": 1, "right": 3, "middle": 2, "back": 8, "forward": 9}
        button_num = button_map.get(button, 1)
        result = await self.executor.execute(f"mousedown {button_num}", take_screenshot=False)
        return result

    # mouse_up - generic mouse button up
    async def mouse_up(
        self, button: Literal["left", "right", "middle", "back", "forward"] = "left"
    ) -> ToolResult:
        """
        Release a mouse button.

        Args:
            button: Mouse button to release
        """
        button_map = {"left": 1, "right": 3, "middle": 2, "back": 8, "forward": 9}
        button_num = button_map.get(button, 1)
        result = await self.executor.execute(f"mouseup {button_num}", take_screenshot=False)
        return result

    # hold_key (Anthropic specific action)
    async def hold_key(self, text: str, duration: float) -> ToolResult:
        """
        Hold a key for a specified duration.

        Args:
            text: The key to hold
            duration: Duration in seconds
        """
        import shlex

        escaped_key = shlex.quote(text)

        # Press the key
        await self.executor.execute(f"keydown {escaped_key}", take_screenshot=False)

        # Wait
        await asyncio.sleep(duration)

        # Release the key
        result = await self.executor.execute(f"keyup {escaped_key}", take_screenshot=False)
        return result
