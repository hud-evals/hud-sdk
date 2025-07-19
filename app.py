#!/usr/bin/env python3
"""
Simple MCP test server for HUD tools.

This server exposes all HUD tools (computer, bash, edit) via MCP for testing.
"""

import asyncio
import logging
import os
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from pydantic import Field

from hud.tools import BashTool, EditTool, HudComputerTool
from hud.tools.edit import Command

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("hud_tools_test", port=8049, log_level="DEBUG", debug=True)

# Initialize tools
computer_tool = HudComputerTool()
bash_tool = BashTool()
edit_tool = EditTool()


# Computer tool wrapper
@mcp.tool()
async def computer(
    action: str = Field(..., description="The action name (click, type, move, etc.)"),
    # Click parameters
    x: int | None = Field(None, description="X coordinate for click/move/scroll actions"),
    y: int | None = Field(None, description="Y coordinate for click/move/scroll actions"),
    button: Literal["left", "right", "middle", "back", "forward"] | None = Field(None, description="Mouse button for click actions"),
    pattern: list[int] | None = Field(None, description="Click pattern for multi-clicks (e.g., [100] for double-click)"),
    # Key/Type parameters
    text: str | None = Field(None, description="Text for type/response actions"),
    keys: list[str] | None = Field(None, description="Keys for press/keydown/keyup actions"),
    enter_after: bool | None = Field(None, description="Whether to press Enter after typing"),
    # Scroll parameters
    scroll_x: int | None = Field(None, description="Horizontal scroll amount (positive = right)"),
    scroll_y: int | None = Field(None, description="Vertical scroll amount (positive = down)"),
    # Move parameters
    offset_x: int | None = Field(None, description="X offset for relative move"),
    offset_y: int | None = Field(None, description="Y offset for relative move"),
    # Drag parameters
    path: list[tuple[int, int]] | None = Field(None, description="Path for drag actions as list of (x, y) coordinates"),
    # Wait parameter
    time: int | None = Field(None, description="Time in milliseconds for wait action"),
    # General parameters
    hold_keys: list[str] | None = Field(None, description="Keys to hold during action"),
    # hold_key specific
    duration: float | None = Field(None, description="Duration in seconds for hold_key action")
) -> list[ImageContent | TextContent]:
    """
    Use this tool to control the computer - click, type, move mouse, take screenshots, etc.
    
    Available actions:
    - screenshot: Take a screenshot
    - click: Click at coordinates with optional button and pattern
    - type: Type text with optional enter
    - move: Move mouse to coordinates
    - scroll: Scroll at position
    - drag: Drag along a path
    - press/keydown/keyup: Keyboard controls
    - wait: Wait for specified time
    - position: Get cursor position
    - response: Return text response
    - hold_key: Hold a key for duration
    - mouse_down/mouse_up: Mouse button controls
    """
    return await computer_tool(
        action=action,
        x=x,
        y=y,
        button=button,
        pattern=pattern,
        text=text,
        keys=keys,
        enter_after=enter_after,
        scroll_x=scroll_x,
        scroll_y=scroll_y,
        offset_x=offset_x,
        offset_y=offset_y,
        path=path,
        time=time,
        hold_keys=hold_keys,
        duration=duration
    )


# Bash tool wrapper
@mcp.tool()
async def bash(
    command: str | None = Field(None, description="The bash command to execute"),
    restart: bool = Field(False, description="Whether to restart the bash session")
) -> list[ImageContent | TextContent]:
    """
    Execute bash commands in a persistent session.
    
    Use restart=True to start a new session if needed.
    """
    result = await bash_tool(command=command, restart=restart)
    
    # Convert ToolResult to content blocks
    blocks = []
    if result.output:
        blocks.append(TextContent(text=result.output, type="text"))
    if result.error:
        blocks.append(TextContent(text=f"Error: {result.error}", type="text"))
    if hasattr(result, 'system') and result.system:
        blocks.append(TextContent(text=f"System: {result.system}", type="text"))
    
    return blocks


# Edit tool wrapper
@mcp.tool()
async def edit(
    command: Command = Field(..., description="The edit command (view, create, str_replace, insert, undo_edit)"),
    path: str = Field(..., description="The file path (must be absolute)"),
    file_text: str | None = Field(None, description="Content for create command"),
    view_range: list[int] | None = Field(None, description="Line range for view command [start, end]"),
    old_str: str | None = Field(None, description="String to replace for str_replace command"),
    new_str: str | None = Field(None, description="Replacement string for str_replace command"),
    insert_line: int | None = Field(None, description="Line number for insert command")
) -> list[ImageContent | TextContent]:
    """
    Edit files with various commands:
    - view: View file contents or directory listings
    - create: Create a new file
    - str_replace: Replace a unique string in a file
    - insert: Insert text at a specific line
    - undo_edit: Undo the last edit
    """
    result = await edit_tool(
        command=command,
        path=path,
        file_text=file_text,
        view_range=view_range,
        old_str=old_str,
        new_str=new_str,
        insert_line=insert_line
    )
    
    # Convert ToolResult to content blocks
    blocks = []
    if result.output:
        blocks.append(TextContent(text=result.output, type="text"))
    if result.error:
        blocks.append(TextContent(text=f"Error: {result.error}", type="text"))
    
    return blocks


def main():
    """Run the MCP test server."""
    logger.info("Starting HUD tools test server...")
    logger.info("Available tools: computer, bash, edit")
    
    # Set environment variables if needed
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"
    
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main() 