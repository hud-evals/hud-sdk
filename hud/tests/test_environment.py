"""Tests for environment logging functionality."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from hud.env.client import Client
from hud.env.environment import Environment
from hud.settings import settings
from hud.types import EnvironmentStatus
from hud.utils.common import FunctionConfig, Observation


class AsyncContextManagerMock:
    """Mock for async context manager."""

    def __init__(self, mock_client):
        self.mock_client = mock_client

    async def __aenter__(self):
        return self.mock_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockClient(Client):
    """Mock client for testing."""

    def __init__(self):
        super().__init__()
        self._invoke = AsyncMock(return_value=({"observation": {}}, b"", b""))
        self._env_id = "test-env-123"

    async def invoke(self, config):
        return await self._invoke(config)

    async def get_status(self):
        return EnvironmentStatus.RUNNING

    async def close(self):
        pass

    @property
    def env_id(self) -> str:
        return self._env_id


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    return MockClient()


@pytest.fixture
def environment(mock_client):
    """Create a test environment."""
    return Environment(
        metadata={"test": "metadata"},  # Required field
        client=mock_client,
        build_data={"test": "build_data"},  # Required field
        url=None,  # Optional
        live_url=None,  # Optional
        task=None,  # Optional
        final_response=None,  # Optional
    )


@pytest.fixture(autouse=True)
def setup_settings():
    """Set up test settings."""
    original_api_key = settings.api_key
    original_base_url = settings.base_url

    # Set test values
    settings.api_key = "test-api-key"
    settings.base_url = "http://test-url"

    yield

    # Restore original values
    settings.api_key = original_api_key
    settings.base_url = original_base_url


@pytest.mark.asyncio
async def test_log_observation_direct(environment):
    """Test that environment.log_observation works correctly."""
    observation = Observation(
        text="test observation",
        start_timestamp=datetime.now(),
        end_timestamp=datetime.now(),
        stdout=b"test stdout",
        stderr=b"test stderr",
        actions=[{"type": "test", "text": "test action"}],
    )

    mock_client = AsyncMock()
    mock_client.post.return_value.status_code = 200

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        await environment.log_observation(observation)

        # Verify the request was made with correct data
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check URL
        expected_url = (
            f"{settings.base_url}/v2/environments/{environment.client.env_id}/log_observation"
        )
        assert args[0] == expected_url

        # Check headers
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["Authorization"] == f"Bearer {settings.api_key}"

        # Check request data
        request_data = kwargs["json"]
        assert request_data == observation.to_json()

        # Check timeout
        assert kwargs["timeout"] == 30.0


@pytest.mark.asyncio
async def test_log_observation_during_step(environment):
    """Test that environment.step logs observations when log_observation=True."""
    # Mock the client's invoke method to return a valid observation
    environment.client._invoke.return_value = (
        {
            "observation": {
                "text": "test observation",
                "screenshot": None,
            }
        },
        b"test stdout",
        b"test stderr",
    )

    mock_client = AsyncMock()
    mock_client.post.return_value.status_code = 200

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        # Call step with log_observation=True
        await environment.step(log_observation=True)

        # Verify the request was made
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check URL
        expected_url = (
            f"{settings.base_url}/v2/environments/{environment.client.env_id}/log_observation"
        )
        assert args[0] == expected_url

        # Check headers
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["Authorization"] == f"Bearer {settings.api_key}"

        # Check request data structure
        request_data = kwargs["json"]
        assert "text" in request_data
        assert "start_timestamp" in request_data
        assert "end_timestamp" in request_data
        assert request_data["stdout"] == "test stdout"
        assert request_data["stderr"] == "test stderr"
        assert request_data["actions"] == []


@pytest.mark.asyncio
async def test_log_score_direct(environment):
    """Test that environment.log_score works correctly."""
    score = 0.95

    mock_client = AsyncMock()
    mock_client.post.return_value.status_code = 200

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        await environment.log_score(score)

        # Verify the request was made with correct data
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check URL
        expected_url = f"{settings.base_url}/v2/environments/{environment.client.env_id}/log_score"
        assert args[0] == expected_url

        # Check headers
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["Authorization"] == f"Bearer {settings.api_key}"

        # Check request data
        assert kwargs["json"] == {"score": score}

        # Check timeout
        assert kwargs["timeout"] == 30.0


@pytest.mark.asyncio
async def test_log_score_during_evaluate(environment):
    """Test that environment.evaluate logs scores when log_score=True."""
    # Mock the client's invoke method to return a score
    environment.client._invoke.return_value = (0.95, b"", b"")

    mock_client = AsyncMock()
    mock_client.post.return_value.status_code = 200

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        # Call evaluate with log_score=True and a test config
        await environment.evaluate(config=FunctionConfig(function="test", args=[]), log_score=True)

        # Verify the request was made
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check URL
        expected_url = f"{settings.base_url}/v2/environments/{environment.client.env_id}/log_score"
        assert args[0] == expected_url

        # Check headers
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["Authorization"] == f"Bearer {settings.api_key}"

        # Check request data
        assert kwargs["json"] == {"score": 0.95}

        # Check timeout
        assert kwargs["timeout"] == 30.0


@pytest.mark.asyncio
async def test_log_observation_error_handling(environment):
    """Test that environment.log_observation handles errors gracefully."""
    observation = Observation(
        text="test observation",
        start_timestamp=datetime.now(),
        end_timestamp=datetime.now(),
    )

    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("Test error")

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        # Should not raise an exception
        await environment.log_observation(observation)


@pytest.mark.asyncio
async def test_log_score_error_handling(environment):
    """Test that environment.log_score handles errors gracefully."""
    score = 0.95

    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("Test error")

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        # Should not raise an exception
        await environment.log_score(score)
