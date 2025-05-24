from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from hud.agent.base import Agent
from hud.adapters import Adapter
from hud.adapters.common.types import ClickAction, Point
from hud.utils.common import Observation


class ConcreteAgent(Agent[Any, dict[str, Any]]):
    """Concrete implementation of Agent for testing."""

    def __init__(self, client: Any = None, adapter: Adapter | None = None):
        super().__init__(client, adapter)
        self.mock_responses = []
        self.call_count = 0

    async def fetch_response(self, observation: Observation) -> tuple[list[dict[str, Any]], bool]:
        """Mock implementation that returns predefined responses."""
        if self.call_count < len(self.mock_responses):
            response = self.mock_responses[self.call_count]
            self.call_count += 1
            return response
        return [], True


class TestAgentBase:
    """Test the base Agent class."""

    @pytest.fixture
    def mock_client(self):
        """Mock client for testing."""
        return MagicMock()

    @pytest.fixture
    def mock_adapter(self):
        """Mock adapter for testing."""
        adapter = MagicMock(spec=Adapter)
        adapter.rescale.return_value = "rescaled_screenshot"
        adapter.adapt_list.return_value = [ClickAction(point=Point(x=100, y=200))]
        return adapter

    @pytest.fixture
    def agent_with_adapter(self, mock_client, mock_adapter):
        """Agent with both client and adapter."""
        return ConcreteAgent(client=mock_client, adapter=mock_adapter)

    @pytest.fixture
    def agent_without_adapter(self, mock_client):
        """Agent with client but no adapter."""
        return ConcreteAgent(client=mock_client, adapter=None)

    def test_init_with_client_and_adapter(self, mock_client, mock_adapter):
        """Test agent initialization with client and adapter."""
        agent = ConcreteAgent(client=mock_client, adapter=mock_adapter)
        assert agent.client == mock_client
        assert agent.adapter == mock_adapter

    def test_init_with_none_values(self):
        """Test agent initialization with None values."""
        agent = ConcreteAgent(client=None, adapter=None)
        assert agent.client is None
        assert agent.adapter is None

    def test_preprocess_without_adapter(self, agent_without_adapter):
        """Test preprocess when no adapter is available."""
        observation = Observation(text="test", screenshot="screenshot_data")
        result = agent_without_adapter.preprocess(observation)

        # Should return original observation unchanged
        assert result == observation
        assert result.text == "test"
        assert result.screenshot == "screenshot_data"

    def test_preprocess_without_screenshot(self, agent_with_adapter):
        """Test preprocess when no screenshot is available."""
        observation = Observation(text="test", screenshot=None)
        result = agent_with_adapter.preprocess(observation)

        # Should return original observation unchanged
        assert result == observation
        assert result.text == "test"
        assert result.screenshot is None

    def test_preprocess_with_adapter_and_screenshot(self, agent_with_adapter, mock_adapter):
        """Test preprocess with adapter and screenshot (covers missing lines 48-55)."""
        observation = Observation(text="test", screenshot="original_screenshot")
        result = agent_with_adapter.preprocess(observation)

        # Should create new observation with rescaled screenshot
        mock_adapter.rescale.assert_called_once_with("original_screenshot")
        assert result.text == "test"
        assert result.screenshot == "rescaled_screenshot"
        # Should be a new object, not the original
        assert result is not observation

    def test_postprocess_without_adapter(self, agent_without_adapter):
        """Test postprocess when no adapter is available (covers missing lines 82-85)."""
        actions = [{"type": "click", "x": 100, "y": 200}]

        with pytest.raises(ValueError, match="Cannot postprocess actions without an adapter"):
            agent_without_adapter.postprocess(actions)

    def test_postprocess_with_adapter(self, agent_with_adapter, mock_adapter):
        """Test postprocess with adapter."""
        actions = [{"type": "click", "x": 100, "y": 200}]
        result = agent_with_adapter.postprocess(actions)

        mock_adapter.adapt_list.assert_called_once_with(actions)
        assert len(result) == 1
        assert isinstance(result[0], ClickAction)

    @pytest.mark.asyncio
    async def test_predict_without_verbose(self, agent_with_adapter):
        """Test predict method without verbose logging."""
        observation = Observation(text="test", screenshot="screenshot")
        agent_with_adapter.mock_responses = [([{"type": "click", "x": 100, "y": 200}], False)]

        actions, done = await agent_with_adapter.predict(observation, verbose=False)

        assert len(actions) == 1
        assert isinstance(actions[0], ClickAction)
        assert done is False

    @pytest.mark.asyncio
    @patch("hud.agent.base.logger")
    async def test_predict_with_verbose_logging(self, mock_logger, agent_with_adapter):
        """Test predict method with verbose logging (covers missing lines 100-116)."""
        observation = Observation(text="test", screenshot="screenshot")
        agent_with_adapter.mock_responses = [([{"type": "click", "x": 100, "y": 200}], True)]

        actions, done = await agent_with_adapter.predict(observation, verbose=True)

        # Verify verbose logging was called
        mock_logger.info.assert_any_call("[hud] Predicting action...")
        mock_logger.info.assert_any_call(
            "[hud] Raw action: %s", [{"type": "click", "x": 100, "y": 200}]
        )

        assert len(actions) == 1
        assert isinstance(actions[0], ClickAction)
        assert done is True

    @pytest.mark.asyncio
    async def test_predict_without_adapter_returns_raw_actions(self, agent_without_adapter):
        """Test predict without adapter returns raw actions."""
        observation = Observation(text="test", screenshot=None)
        raw_actions = [{"type": "click", "x": 100, "y": 200}]
        agent_without_adapter.mock_responses = [(raw_actions, True)]

        actions, done = await agent_without_adapter.predict(observation, verbose=False)

        # Should return raw actions, not processed ones
        assert actions == raw_actions
        assert done is True

    @pytest.mark.asyncio
    async def test_predict_with_empty_actions(self, agent_with_adapter):
        """Test predict when fetch_response returns empty actions."""
        observation = Observation(text="test", screenshot="screenshot")
        agent_with_adapter.mock_responses = [([], True)]

        actions, done = await agent_with_adapter.predict(observation, verbose=False)

        # Should return empty actions without calling adapter
        assert actions == []
        assert done is True

    @pytest.mark.asyncio
    async def test_predict_full_pipeline(self, agent_with_adapter, mock_adapter):
        """Test the complete predict pipeline with all stages."""
        # Set up observation with screenshot that will be rescaled
        observation = Observation(text="test input", screenshot="original_screenshot")
        raw_actions = [{"type": "click", "x": 150, "y": 250}]
        agent_with_adapter.mock_responses = [(raw_actions, False)]

        actions, done = await agent_with_adapter.predict(observation, verbose=True)

        # Verify all stages were called
        # Stage 1: Preprocessing
        mock_adapter.rescale.assert_called_once_with("original_screenshot")

        # Stage 3: Postprocessing
        mock_adapter.adapt_list.assert_called_once_with(raw_actions)

        assert len(actions) == 1
        assert isinstance(actions[0], ClickAction)
        assert done is False

    @pytest.mark.asyncio
    async def test_predict_integration_without_screenshot(self, agent_with_adapter):
        """Test predict integration when observation has no screenshot."""
        observation = Observation(text="test input", screenshot=None)
        raw_actions = [{"type": "response", "text": "Task completed"}]
        agent_with_adapter.mock_responses = [(raw_actions, True)]

        actions, done = await agent_with_adapter.predict(observation, verbose=False)

        assert len(actions) == 1
        assert done is True
