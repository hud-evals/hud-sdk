"""Tests for telemetry exporter functions."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from hud.settings import settings
from hud.telemetry.exporter import log_observation, log_score
from hud.utils.common import Observation


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


class AsyncContextManagerMock:
    """Mock for async context manager."""

    def __init__(self, mock_client):
        self.mock_client = mock_client

    async def __aenter__(self):
        return self.mock_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
async def test_log_observation():
    """Test that log_observation sends the correct data to the endpoint."""
    env_id = "test-env-123"
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
        await log_observation(env_id, observation)

        # Verify the request was made with correct data
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check URL
        assert args[0] == f"{settings.base_url}/v2/environments/{env_id}/log_observation"

        # Check headers
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["Authorization"] == f"Bearer {settings.api_key}"

        # Check request data
        request_data = kwargs["json"]
        assert request_data == observation.to_json()

        # Check timeout
        assert kwargs["timeout"] == 30.0


@pytest.mark.asyncio
async def test_log_observation_error_handling():
    """Test that log_observation handles errors gracefully."""
    env_id = "test-env-123"
    observation = Observation(
        text="test observation",
        start_timestamp=datetime.now(),
        end_timestamp=datetime.now(),
    )

    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("Test error")

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        # Should not raise an exception
        await log_observation(env_id, observation)


@pytest.mark.asyncio
async def test_log_score():
    """Test that log_score sends the correct data to the endpoint."""
    env_id = "test-env-123"
    score = 0.95

    mock_client = AsyncMock()
    mock_client.post.return_value.status_code = 200

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        await log_score(env_id, score)

        # Verify the request was made with correct data
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check URL
        assert args[0] == f"{settings.base_url}/v2/environments/{env_id}/log_score"

        # Check headers
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["Authorization"] == f"Bearer {settings.api_key}"

        # Check request data
        assert kwargs["json"] == {"score": score}

        # Check timeout
        assert kwargs["timeout"] == 30.0


@pytest.mark.asyncio
async def test_log_score_error_handling():
    """Test that log_score handles errors gracefully."""
    env_id = "test-env-123"
    score = 0.95

    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("Test error")

    with patch("httpx.AsyncClient", return_value=AsyncContextManagerMock(mock_client)):
        # Should not raise an exception
        await log_score(env_id, score)
