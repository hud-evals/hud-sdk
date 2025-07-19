import asyncio
import logging
from typing import Any, cast

from pydantic import Field
from anthropic.types.beta import BetaToolComputerUse20250124Param
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ImageContent, TextContent

from .hud import HudComputerTool
from ..base import tool_result_to_content_blocks, ToolResult

logger = logging.getLogger(__name__)


class AnthropicComputerTool(HudComputerTool):
    """
    Anthropic Computer Use tool that converts Anthropic's action format
    to HudComputerTool calls.
    """
    
    name: str = "computer"
    api_type: str = "computer_20250124"
    
    def __init__(self, width: int = 1024, height: int = 768, display_num: int | None = None):
        """
        Initialize with Anthropic's default dimensions.
        
        Args:
            width: Screen width (default: 1024 for Anthropic)
            height: Screen height (default: 768 for Anthropic)
            display_num: X display number
        """
        super().__init__(width=width, height=height, display_num=display_num)
    
    def to_params(self) -> BetaToolComputerUse20250124Param:
        """Convert to Anthropic tool parameters."""
        return cast(
            BetaToolComputerUse20250124Param,
            {
                "type": self.api_type,
                "name": self.name,
                "display_width_px": self.width,
                "display_height_px": self.height,
            }
        )
    
    async def __call__(
        self,
        *,
        action: str = Field(..., description="The action to perform on the computer"),
        coordinate: list[int] | tuple[int, int] | None = Field(None, description="The coordinate to interact with on the computer [x, y]"),
        text: str | None = Field(None, description="The text to type on the computer or key to press"),
        start_coordinate: list[int] | tuple[int, int] | None = Field(None, description="The starting coordinate for drag actions [x, y]"),
        scroll_direction: str | None = Field(None, description="The direction to scroll (up, down, left, right)"),
        scroll_amount: int | None = Field(None, description="The amount to scroll"),
        duration: int | float | None = Field(None, description="The duration of the action in seconds"),
        take_screenshot_on_click: bool = Field(True, description="Whether to take a screenshot after clicking")
    ) -> list[ImageContent | TextContent]:
        """
        Handle Anthropic Computer Use API calls.
        
        This converts Anthropic's action format to HudComputerTool's format.
        
        Returns:
            List of MCP content blocks
        """
        logger.info(f"AnthropicComputerTool received action: {action}")
        
        # Convert lists to tuples if needed
        coord_tuple = None
        if coordinate:
            coord_tuple = tuple(coordinate) if isinstance(coordinate, list) else coordinate
        
        start_coord_tuple = None
        if start_coordinate:
            start_coord_tuple = tuple(start_coordinate) if isinstance(start_coordinate, list) else start_coordinate
        
        # Map Anthropic actions to HudComputerTool actions
        if action == "screenshot":
            result = await self.screenshot()
            
        elif action == "left_click" or action == "click":
            if coord_tuple and len(coord_tuple) >= 2:
                result = await self.click(x=coord_tuple[0], y=coord_tuple[1])
            else:
                result = await self.click()
            
        elif action == "double_click":
            if coord_tuple and len(coord_tuple) >= 2:
                # Use pattern for double-click
                result = await self.click(x=coord_tuple[0], y=coord_tuple[1], pattern=[100])
            else:
                result = await self.click(pattern=[100])
        
        elif action == "triple_click":
            if coord_tuple and len(coord_tuple) >= 2:
                # Use pattern for triple-click
                result = await self.click(x=coord_tuple[0], y=coord_tuple[1], pattern=[100, 100])
            else:
                result = await self.click(pattern=[100, 100])
            
        elif action == "right_click":
            if coord_tuple and len(coord_tuple) >= 2:
                result = await self.click(x=coord_tuple[0], y=coord_tuple[1], button="right")
            else:
                result = await self.click(button="right")
            
        elif action == "middle_click":
            if coord_tuple and len(coord_tuple) >= 2:
                result = await self.click(x=coord_tuple[0], y=coord_tuple[1], button="middle")
            else:
                result = await self.click(button="middle")
            
        elif action == "mouse_move" or action == "move":
            if coord_tuple and len(coord_tuple) >= 2:
                result = await self.move(x=coord_tuple[0], y=coord_tuple[1])
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="coordinate is required for mouse_move"))
            
        elif action == "type":
            if text:
                result = await self.type(text=text)
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for type"))
            
        elif action == "key":
            if text:
                # Anthropic sends single key or combo like "ctrl+a"
                # The key action expects the raw key string, not a list
                result = await self.press(keys=[text])
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for key"))
            
        elif action == "scroll":
            # Original implementation validates scroll_direction and scroll_amount
            if scroll_direction not in ["up", "down", "left", "right"]:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="scroll_direction must be 'up', 'down', 'left', or 'right'"))
            
            if scroll_amount is None or scroll_amount < 0:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="scroll_amount must be a non-negative int"))
            
            # Convert direction to scroll amounts
            scroll_x = None
            scroll_y = None
            if scroll_direction == "down":
                scroll_y = scroll_amount
            elif scroll_direction == "up":
                scroll_y = -scroll_amount
            elif scroll_direction == "right":
                scroll_x = scroll_amount
            elif scroll_direction == "left":
                scroll_x = -scroll_amount
            
            if coord_tuple and len(coord_tuple) >= 2:
                result = await self.scroll(x=coord_tuple[0], y=coord_tuple[1], scroll_x=scroll_x, scroll_y=scroll_y)
            else:
                result = await self.scroll(scroll_x=scroll_x, scroll_y=scroll_y)
            
        elif action == "left_click_drag" or action == "drag":
            # Anthropic sends drag with start and end coordinates
            if coord_tuple and len(coord_tuple) >= 2:
                if start_coord_tuple and len(start_coord_tuple) >= 2:
                    # Full drag path
                    path = [(start_coord_tuple[0], start_coord_tuple[1]), (coord_tuple[0], coord_tuple[1])]
                    result = await self.drag(path=path)
                else:
                    # Just end coordinate, drag from current position
                    # Original spec allows this
                    current_pos = [(0, 0), (coord_tuple[0], coord_tuple[1])]  # Simplified
                    result = await self.drag(path=current_pos)
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="coordinate is required for left_click_drag"))
            
        elif action == "wait":
            # Original spec expects duration in seconds
            if duration is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration is required for wait"))
            if duration < 0:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration must be non-negative"))
            if duration > 100:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration is too long"))
            
            # Convert seconds to milliseconds for HudComputerTool
            result = await self.wait(time=int(duration * 1000))
            
        elif action == "hold_key":
            # Original spec has hold_key action
            if text is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for hold_key"))
            if duration is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration is required for hold_key"))
            if duration < 0:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration must be non-negative"))
            if duration > 100:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration is too long"))
                
            # Hold key action
            result = await self.hold_key(text=text, duration=duration)
            
        elif action == "left_mouse_down":
            # These don't accept coordinates in original spec
            if coord_tuple is not None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="coordinate is not accepted for left_mouse_down"))
            # Use generic mouse_down method
            result = await self.mouse_down(button="left")
            
        elif action == "left_mouse_up":
            # These don't accept coordinates in original spec
            if coord_tuple is not None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="coordinate is not accepted for left_mouse_up"))
            # Use generic mouse_up method
            result = await self.mouse_up(button="left")
            
        elif action == "cursor_position":
            result = await self.position()
            
        else:
            # Unknown action
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Invalid action: {action}"))
        
        # Handle screenshot for actions that need it
        screenshot_actions = {
            "screenshot", "left_click", "click", "double_click", "triple_click",
            "right_click", "middle_click", "mouse_move", "move", "type", "key",
            "scroll", "left_click_drag", "drag", "wait", "hold_key",
            "left_mouse_down", "left_mouse_up"
        }
        
        if action in screenshot_actions and action != "screenshot" and take_screenshot_on_click:
            # Add screenshot to result if not already present
            if isinstance(result, ToolResult) and not result.base64_image:
                screenshot_base64 = await self.executor.screenshot()
                if screenshot_base64:
                    result = ToolResult(
                        output=result.output,
                        error=result.error,
                        base64_image=screenshot_base64
                    )
        
        # Convert to content blocks
        return tool_result_to_content_blocks(result) 