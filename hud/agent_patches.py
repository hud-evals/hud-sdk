"""Client-side patches for handling known server issues gracefully."""

from __future__ import annotations

import logging
from typing import Any

# MCPToolResult is used in the patched function but imported dynamically

logger = logging.getLogger(__name__)


def patch_mcp_client_call_tool() -> None:
    """
    Patch MCP client's call_tool method to include telemetry for HUD integration.
    """
    try:
        from mcp import types

        from hud.clients.fastmcp import FastMCPHUDClient
        from hud.types import MCPToolResult
    except ImportError:
        logging.warning("MCP packages not available for patching")
        return

    def patched_call_tool(self: Any, name: str, arguments: dict[str, Any]) -> Any:
        """
        Patched version of call_tool with telemetry integration.
        """
        try:
            # Get the original result
            if hasattr(self, "_original_call_tool"):
                original_result = self._original_call_tool(name, arguments)
            else:
                # Fallback to calling the method directly
                original_result = super(type(self), self).call_tool(name, arguments)

            # Convert result to text for telemetry
            result_text = getattr(original_result, "text", str(original_result))

            # Log the tool call for debugging
            logging.debug(
                "Tool call completed: %s with args %s -> %s",
                name,
                arguments,
                result_text[:100] + "..." if len(result_text) > 100 else result_text,
            )

            # Create HUD-compatible result
            if isinstance(original_result, types.TextContent):
                return MCPToolResult(
                    content=[original_result],
                    tool_call_id=getattr(arguments, "tool_call_id", None),
                )
            else:
                # For other types, wrap in TextContent
                return MCPToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=result_text,
                        )
                    ],
                    tool_call_id=getattr(arguments, "tool_call_id", None),
                )

        except Exception as e:
            logging.error("Error in patched call_tool: %s", str(e))
            # Return original result on error
            if hasattr(self, "_original_call_tool"):
                return self._original_call_tool(name, arguments)
            else:
                return super(type(self), self).call_tool(name, arguments)

    # Apply the patch to FastMCP clients
    try:
        # Store original method
        if not hasattr(FastMCPHUDClient, "_original_call_tool"):
            FastMCPHUDClient._original_call_tool = FastMCPHUDClient.call_tool

        # Replace with patched version
        FastMCPHUDClient.call_tool = patched_call_tool
        logging.info("Successfully patched FastMCPHUDClient.call_tool")

    except Exception as e:
        logging.error("Failed to patch FastMCPHUDClient.call_tool: %s", str(e))


def apply_all_patches() -> None:
    """Apply all available patches for MCP integration."""
    patch_mcp_client_call_tool()
