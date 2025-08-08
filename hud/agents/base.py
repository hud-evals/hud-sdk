"""Base classes for MCP agents."""

from __future__ import annotations

from abc import ABC
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel

from ..agent import MCPAgent
from ..types import AgentResult

if TYPE_CHECKING:
    from ..datasets import TaskConfig


class ModelResponse(BaseModel):
    """Response from a model."""
    content: str
    isError: bool = False
    error_message: str | None = None
    
    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict."""
        return {
            "content": self.content,
            "isError": self.isError,
            "error_message": self.error_message
        }


class BaseMCPAgent(MCPAgent):
    """
    Base class for MCP agents with common functionality.
    
    This class extends MCPAgent with provider-specific implementations
    and common helper methods.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base MCP agent."""
        super().__init__(**kwargs)
    
    async def get_model_response(self, messages: list[Any], **kwargs: Any) -> ModelResponse:
        """Get response from the model. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_model_response")
    
    async def execute_tools(self, *, tool_calls: list[Any], **kwargs: Any) -> list[AgentResult]:
        """Execute tool calls. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute_tools")