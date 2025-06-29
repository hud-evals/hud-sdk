"""Tests for telemetry exporter functions."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
import json
import logging
from typing import Any, Dict

import httpx
from pytest_httpx import HTTPXMock

from hud.settings import settings
from hud.telemetry.async_logger import AsyncLogger
from hud.utils.common import Observation

logger = logging.getLogger(__name__)


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


@pytest.fixture
def observation() -> Dict[str, Any]:
    return {
        "type": "text",
        "text": "test observation",
        "timestamp": datetime.now().timestamp(),
    }


@pytest.mark.asyncio
async def test_log_observation(httpx_mock: HTTPXMock, observation: Dict[str, Any]):
    """Test that log_observation sends the correct data to the endpoint."""
    env_id = "test-env"
    httpx_mock.add_response(status_code=200)

    logger = AsyncLogger.get_instance()
    await logger.log_observation(env_id, Observation(**observation))

    request = httpx_mock.get_request()
    assert request is not None
    assert request.method == "POST"
    assert request.url == f"{settings.base_url}/v2/environments/{env_id}/log_observation"
    assert request.headers["Authorization"] == f"Bearer {settings.api_key}"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == observation


@pytest.mark.asyncio
async def test_log_observation_error_handling(httpx_mock: HTTPXMock, observation: Dict[str, Any]):
    """Test that log_observation handles errors gracefully."""
    env_id = "test-env"
    httpx_mock.add_response(status_code=500, text="Internal Server Error")

    logger = AsyncLogger.get_instance()
    await logger.log_observation(env_id, Observation(**observation))


@pytest.mark.asyncio
async def test_log_score(httpx_mock: HTTPXMock):
    """Test that log_score sends the correct data to the endpoint."""
    env_id = "test-env"
    score = 0.5
    httpx_mock.add_response(status_code=200)

    logger = AsyncLogger.get_instance()
    await logger.log_score(env_id, score)

    request = httpx_mock.get_request()
    assert request is not None
    assert request.method == "POST"
    assert request.url == f"{settings.base_url}/v2/environments/{env_id}/log_score"
    assert request.headers["Authorization"] == f"Bearer {settings.api_key}"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {"score": score}


@pytest.mark.asyncio
async def test_log_score_error_handling(httpx_mock: HTTPXMock):
    """Test that log_score handles errors gracefully."""
    env_id = "test-env"
    score = 0.5
    httpx_mock.add_response(status_code=500, text="Internal Server Error")

    logger = AsyncLogger.get_instance()
    await logger.log_score(env_id, score)
