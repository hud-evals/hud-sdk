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

from hud.agent.claude import ClaudeAgent
from hud.agent.operator import OperatorAgent
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
def claude_agent(mock_anthropic_client: AsyncMock, mock_adapter: MagicMock) -> ClaudeAgent:
    """Create a ClaudeAgent instance with mocked dependencies."""
    return ClaudeAgent(client=mock_anthropic_client, adapter=mock_adapter)


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    return client


@pytest.fixture
def operator_agent(mock_openai_client: AsyncMock, mock_adapter: MagicMock) -> OperatorAgent:
    """Create an OperatorAgent instance with mocked dependencies."""
    return OperatorAgent(client=mock_openai_client, adapter=mock_adapter)


@pytest.mark.asyncio
async def test_claude_fetch_response_text_only(
    claude_agent: ClaudeAgent,
    mock_anthropic_client: AsyncMock,
) -> None:
    """Test fetch_response with text-only observation."""
    observation = Observation(text="Test prompt", screenshot=None)

    mock_response = AsyncMock(spec=Message)
    text_block = MagicMock(spec=BetaTextBlockParam)
    text_block.type = "text"
    text_block.text = "This is a test response"
    mock_response.content = [text_block]
    mock_anthropic_client.beta.messages.create.return_value = mock_response

    actions, done, logs = await claude_agent.fetch_response(observation)

    mock_anthropic_client.beta.messages.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {"action": "response", "text": "This is a test response"}
    assert done is True
    assert logs == [mock_response.model_dump()]


@pytest.mark.asyncio
async def test_claude_fetch_response_with_tool_use(
    claude_agent: ClaudeAgent,
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
    mock_anthropic_client.beta.messages.create.return_value = mock_response

    actions, done, logs = await claude_agent.fetch_response(observation)

    mock_anthropic_client.beta.messages.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {"action": "click", "coordinates": {"x": 100, "y": 200}}
    assert done is False
    assert claude_agent.pending_computer_use_tool_id == "tool_123"
    assert logs == [mock_response.model_dump()]


@pytest.mark.asyncio
async def test_claude_fetch_response_with_screenshot_and_pending_tool(
    claude_agent: ClaudeAgent,
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
    mock_anthropic_client.beta.messages.create.return_value = mock_response

    actions, done, logs = await claude_agent.fetch_response(observation)

    mock_anthropic_client.beta.messages.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {"action": "response", "text": "Task completed successfully"}
    assert done is True
    assert claude_agent.pending_computer_use_tool_id is None
    assert logs == [mock_response.model_dump()]


@pytest.mark.asyncio
async def test_operator_fetch_response_text_only(
    operator_agent: OperatorAgent,
    mock_openai_client: AsyncMock,
) -> None:
    """Test fetch_response with text-only observation for OperatorAgent."""
    observation = Observation(text="Test prompt", screenshot=None)

    mock_response = AsyncMock(spec=Response)
    mock_response.id = "resp_123"
    mock_message = MagicMock(spec=ResponseOutputMessage)
    mock_message.type = "message"
    mock_text = MagicMock(spec=ResponseOutputText)
    mock_text.text = "This is a test response"
    mock_message.content = [mock_text]
    mock_response.output = [mock_message]
    mock_openai_client.responses.create.return_value = mock_response

    actions, done, logs = await operator_agent.fetch_response(observation)

    mock_openai_client.responses.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {"type": "response", "text": "This is a test response"}
    assert done is True
    assert logs == [mock_response.model_dump()]


@pytest.mark.asyncio
async def test_operator_fetch_response_with_computer_call(
    operator_agent: OperatorAgent,
    mock_openai_client: AsyncMock,
) -> None:
    """Test fetch_response when OperatorAgent uses the computer tool."""
    observation = Observation(text="Click the button", screenshot="base64_screenshot_data")

    mock_response = AsyncMock(spec=Response)
    mock_response.id = "resp_123"
    mock_computer_call = MagicMock(spec=ResponseComputerToolCall)
    mock_computer_call.type = "computer_call"
    mock_computer_call.call_id = "call_123"
    mock_computer_call.pending_safety_checks = []
    mock_action = MagicMock()
    mock_action.model_dump.return_value = {"type": "click", "coordinates": {"x": 100, "y": 200}}
    mock_computer_call.action = mock_action
    mock_response.output = [mock_computer_call]
    mock_openai_client.responses.create.return_value = mock_response

    actions, done, logs = await operator_agent.fetch_response(observation)

    mock_openai_client.responses.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {"type": "click", "coordinates": {"x": 100, "y": 200}}
    assert done is False
    assert operator_agent.pending_call_id == "call_123"
    assert logs == [mock_response.model_dump()]


@pytest.mark.asyncio
async def test_operator_fetch_response_with_screenshot_followup(
    operator_agent: OperatorAgent,
    mock_openai_client: AsyncMock,
) -> None:
    """Test fetch_response with a screenshot when there's a pending call."""
    operator_agent.last_response_id = "resp_123"
    operator_agent.pending_call_id = "call_123"
    observation = Observation(text=None, screenshot="base64_screenshot_data")

    mock_response = AsyncMock(spec=Response)
    mock_response.id = "resp_124"
    mock_message = MagicMock(spec=ResponseOutputMessage)
    mock_message.type = "message"
    mock_text = MagicMock(spec=ResponseOutputText)
    mock_text.text = "Task completed successfully"
    mock_message.content = [mock_text]
    mock_response.output = [mock_message]
    mock_openai_client.responses.create.return_value = mock_response

    actions, done, logs = await operator_agent.fetch_response(observation)

    mock_openai_client.responses.create.assert_called_once()
    assert len(actions) == 1
    assert actions[0] == {"type": "response", "text": "Task completed successfully"}
    assert done is True
    assert logs == [mock_response.model_dump()]
