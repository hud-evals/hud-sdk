import logging
from typing import Any, cast, Literal

from pydantic import Field
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ImageContent, TextContent

from .hud import HudComputerTool
from ..base import tool_result_to_content_blocks, ToolResult

logger = logging.getLogger(__name__)


class OpenAIComputerTool(HudComputerTool):
    """
    OpenAI Computer Use tool that converts OpenAI's action format
    to HudComputerTool calls.
    """
    
    def __init__(self, width: int = 1024, height: int = 768, display_num: int | None = None):
        """
        Initialize with OpenAI's default dimensions.
        
        Args:
            width: Screen width (default: 1024 for OpenAI)
            height: Screen height (default: 768 for OpenAI)
            display_num: X display number
        """
        super().__init__(width=width, height=height, display_num=display_num)
    
    async def __call__(
        self,
        *,
        type: str = Field(..., description="The action type to perform"),
        # Coordinate parameters
        x: int | None = Field(None, description="X coordinate for click/move/scroll actions"),
        y: int | None = Field(None, description="Y coordinate for click/move/scroll actions"),
        # Button parameter
        button: str | None = Field(None, description="Mouse button for click actions (left, right, middle, wheel)"),
        # Text parameter
        text: str | None = Field(None, description="Text to type or response text"),
        # Scroll parameters
        scroll_x: int | None = Field(None, description="Horizontal scroll amount"),
        scroll_y: int | None = Field(None, description="Vertical scroll amount"),
        # Wait parameter
        ms: int | None = Field(None, description="Time to wait in milliseconds"),
        # Key press parameter
        keys: list[str] | None = Field(None, description="Keys to press"),
        # Drag parameter
        path: list[dict[str, int]] | None = Field(None, description="Path for drag actions as list of {x, y} dicts"),
        # Custom action parameter
        action: str | None = Field(None, description="Custom action name")
    ) -> list[ImageContent | TextContent]:
        """
        Handle OpenAI Computer Use API calls.
        
        This converts OpenAI's action format (based on OperatorAdapter) to HudComputerTool's format.
        
        Returns:
            List of MCP content blocks
        """
        logger.info(f"OpenAIComputerTool received type: {type}")
        
        # Map button names
        button_map = {"wheel": "middle"}
        if button:
            button = button_map.get(button, button)
        
        # Process based on action type
        if type == "screenshot":
            result = await self.screenshot()
            
        elif type == "click":
            if x is not None and y is not None:
                # Cast button to proper literal type
                button_literal = cast(Literal["left", "right", "middle", "back", "forward"], button or "left")
                result = await self.click(x=x, y=y, button=button_literal)
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x and y coordinates required for click"))
            
        elif type == "double_click":
            if x is not None and y is not None:
                # Use pattern for double-click
                result = await self.click(x=x, y=y, button="left", pattern=[100])
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x and y coordinates required for double_click"))
            
        elif type == "scroll":
            if x is None or y is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x and y coordinates required for scroll"))
            
            # scroll_x and scroll_y default to 0 if not provided
            result = await self.scroll(
                x=x, 
                y=y, 
                scroll_x=scroll_x or 0, 
                scroll_y=scroll_y or 0
            )
            
        elif type == "type":
            if text is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for type"))
            result = await self.type(text=text, enter_after=False)
            
        elif type == "wait":
            wait_time = ms or 1000  # Default to 1 second
            result = await self.wait(time=wait_time)
            
        elif type == "move":
            if x is not None and y is not None:
                result = await self.move(x=x, y=y)
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x and y coordinates required for move"))
            
        elif type == "keypress":
            if keys is None or len(keys) == 0:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="keys is required for keypress"))
            
            # Map common key names
            key_map = {
                "return": "Return",
                "arrowup": "Up",
                "arrowdown": "Down",
                "arrowleft": "Left",
                "arrowright": "Right",
                "cmd": "ctrl",  # Map cmd to ctrl
                "super": "Super_L",  # Map super to Win key
            }
            
            mapped_keys = []
            for key in keys:
                mapped_key = key_map.get(key.lower(), key)
                mapped_keys.append(mapped_key)
            
            result = await self.press(keys=mapped_keys)
            
        elif type == "drag":
            if path is None or len(path) < 2:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="path with at least 2 points required for drag"))
            
            # Convert path from list of dicts to list of tuples
            drag_path = []
            for point in path:
                if "x" in point and "y" in point:
                    drag_path.append((point["x"], point["y"]))
                else:
                    raise McpError(ErrorData(code=INVALID_PARAMS, message="Each point in path must have x and y"))
            
            result = await self.drag(path=drag_path)
            
        elif type == "response":
            if text is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for response"))
            # Response returns content blocks directly
            return [TextContent(text=text, type="text")]
            
        elif type == "custom":
            # For custom actions, we just return an error since HudComputerTool doesn't support them
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Custom action not supported: {action}"))
            
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Invalid action type: {type}"))
        
        # Handle screenshot for actions that need it
        screenshot_actions = {
            "screenshot", "click", "double_click", "scroll", "type", 
            "move", "keypress", "drag", "wait"
        }
        
        if type in screenshot_actions and type != "screenshot":
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