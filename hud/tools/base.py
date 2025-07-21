from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from mcp.types import ImageContent, TextContent


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None
    system: str | None = None

    def __bool__(self) -> bool:
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: ToolResult) -> ToolResult:
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ) -> str | None:
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs: Any) -> ToolResult:
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


# Legacy alias for backward compatibility
CLIResult = ToolResult


class ToolError(Exception):
    """An error raised by a tool."""


# Legacy alias for backward compatibility
CLIError = ToolError


def tool_result_to_content_blocks(result: ToolResult) -> list[ImageContent | TextContent]:
    """Convert a ToolResult to MCP content blocks."""
    blocks = []

    if result.output:
        blocks.append(TextContent(text=result.output, type="text"))
    if result.error:
        blocks.append(TextContent(text=result.error, type="text"))
    if result.base64_image:
        blocks.append(ImageContent(data=result.base64_image, mimeType="image/png", type="image"))
    return blocks
