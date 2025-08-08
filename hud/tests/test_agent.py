from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import Message
from anthropic.types.beta import (
    BetaTextBlockParam,
    BetaToolUseBlockParam,
)
from openai.types.responses import (
    Response,
    ResponseComputerToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from hud.agents.claude import ClaudeMCPAgent
from hud.utils.common import Observation


@pytest.fixture
def mock_anthropic_client() -> AsyncMock:
    """Mock Anthropic client for testing."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Mock adapter for testing."""
    adapter = MagicMock()
    adapter.agent_width = 1024
    adapter.agent_height = 768
    return adapter


@pytest.fixture
def claude_agent(mock_anthropic_client: AsyncMock, mock_adapter: MagicMock) -> ClaudeMCPAgent:
    """Create a ClaudeMCPAgent instance with mocked dependencies."""
    return ClaudeMCPAgent(client=mock_anthropic_client, adapter=mock_adapter)


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    return client





@pytest.mark.asyncio
async def test_claude_fetch_response_text_only(
    claude_agent: ClaudeMCPAgent,
    mock_anthropic_client: AsyncMock,
) -> None:
    """Test fetch_response with text-only observation."""
    observation = Observation(text="Test prompt", screenshot=None)

    mock_response = AsyncMock(spec=Message)
    text_block = MagicMock(spec=BetaTextBlockParam)
    text_block.type = "text"
    text_block.text = "This is a test response"
    mock_response.content = [text_block]
    mock_response.model_dump.return_value = {
        "content": [{"type": "text", "text": "This is a test response"}]
    }
    mock_anthropic_client.beta.messages.create.return_value = mock_response

    actions, done = await claude_agent.fetch_response(observation)

    mock_anthropic_client.beta.messages.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {
        "action": "response",
        "text": "This is a test response",
        "reasoning": "This is a test response",
        "logs": {"content": [{"type": "text", "text": "This is a test response"}]},
    }
    assert done is True


@pytest.mark.asyncio
async def test_claude_fetch_response_with_tool_use(
    claude_agent: ClaudeMCPAgent,
    mock_anthropic_client: AsyncMock,
) -> None:
    """Test fetch_response when Claude uses the computer tool."""
    observation = Observation(text="Click the button", screenshot="base64_screenshot_data")

    mock_response = AsyncMock(spec=Message)
    tool_block = MagicMock(spec=BetaToolUseBlockParam)
    tool_block.type = "tool_use"
    tool_block.name = "computer"
    tool_block.id = "tool_123"
    tool_block.input = {"action": "click", "coordinates": {"x": 100, "y": 200}}
    mock_response.content = [tool_block]
    mock_response.model_dump.return_value = {
        "content": [
            {
                "type": "tool_use",
                "name": "computer",
                "id": "tool_123",
                "input": {"action": "click", "coordinates": {"x": 100, "y": 200}},
            }
        ]
    }
    mock_anthropic_client.beta.messages.create.return_value = mock_response

    actions, done = await claude_agent.fetch_response(observation)

    mock_anthropic_client.beta.messages.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {
        "action": "click",
        "coordinates": {"x": 100, "y": 200},
        "reasoning": "",
        "logs": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "computer",
                    "id": "tool_123",
                    "input": {"action": "click", "coordinates": {"x": 100, "y": 200}},
                }
            ]
        },
    }
    assert done is False
    assert claude_agent.pending_computer_use_tool_id == "tool_123"


@pytest.mark.asyncio
async def test_claude_fetch_response_with_screenshot_and_pending_tool(
    claude_agent: ClaudeMCPAgent,
    mock_anthropic_client: AsyncMock,
) -> None:
    """Test fetch_response with a screenshot when there's a pending tool use."""
    claude_agent.pending_computer_use_tool_id = "previous_tool_123"
    observation = Observation(text=None, screenshot="base64_screenshot_data")

    mock_response = AsyncMock(spec=Message)
    text_block = MagicMock(spec=BetaTextBlockParam)
    text_block.type = "text"
    text_block.text = "Task completed successfully"
    mock_response.content = [text_block]
    mock_response.model_dump.return_value = {
        "content": [{"type": "text", "text": "Task completed successfully"}]
    }
    mock_anthropic_client.beta.messages.create.return_value = mock_response

    actions, done = await claude_agent.fetch_response(observation)

    mock_anthropic_client.beta.messages.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {
        "action": "response",
        "text": "Task completed successfully",
        "reasoning": "Task completed successfully",
        "logs": {"content": [{"type": "text", "text": "Task completed successfully"}]},
    }
    assert done is True
    assert claude_agent.pending_computer_use_tool_id is None



