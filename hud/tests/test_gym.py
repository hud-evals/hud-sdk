from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ConfigDict

from hud.env.client import Client
from hud.env.environment import Environment
from hud.gym import make
from hud.job import Job
from hud.task import Task
from hud.types import CustomGym, EnvironmentStatus
from hud.utils.config import FunctionConfig


class MockClient(Client):
    """Mock client for testing."""

    model_config = ConfigDict(extra="allow")

    async def invoke(self, config: FunctionConfig) -> tuple[Any, bytes | None, bytes | None]:
        if config.function == "step":
            return {"observation": {"text": "test"}}, None, None
        return {}, None, None

    async def get_status(self) -> EnvironmentStatus:
        return EnvironmentStatus.RUNNING

    async def close(self) -> None:
        pass

    def set_source_path(self, path: Path) -> None:
        pass

    def __init__(self):
        super().__init__()
        self.set_source_path = MagicMock()
        self.reset = AsyncMock()
        self.step = AsyncMock()
        self.evaluate = AsyncMock()
        self._invoke = AsyncMock(return_value=({}, None, None))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "custom_local_gym",
            "env_src": CustomGym(
                dockerfile="test.Dockerfile",
                location="local",
                controller_source_dir="/path/to/source",
            ),
            "mock_path": "hud.gym.LocalDockerClient.create",
            "expected_create_args": ("test.Dockerfile",),
            "expected_source_path": "/path/to/source",
            "config": [FunctionConfig(function="test", args=[])],
        },
        {
            "name": "custom_remote_gym",
            "env_src": Task(
                id="test-task-1",
                prompt="Test Task",
                gym=CustomGym(
                    dockerfile="test.Dockerfile",
                    location="remote",
                    controller_source_dir="/path/to/source",
                ),
            ),
            "mock_path": "hud.gym.RemoteDockerClient.create",
            "expected_create_args": {
                "dockerfile": "test.Dockerfile",
                "job_id": None,
                "task_id": "test-task-1",
                "metadata": {},
            },
            "expected_source_path": "/path/to/source",
        },
        {
            "name": "preconfigured_gym",
            "env_src": "qa",
            "mock_path": "hud.gym.RemoteClient.create",
            "expected_create_args": {
                "gym_id": "true-gym-id",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
            "mock_get_gym_id": True,
        },
    ],
)
async def test_make_gym(mocker, test_case):
    """Test creating environments with different gym types."""
    mock_client = MockClient()
    mock_build_data = {"image": "test-image"}
    mock_create = mocker.patch(test_case["mock_path"], new_callable=AsyncMock)
    mock_create.return_value = (mock_client, mock_build_data)

    if test_case.get("mock_get_gym_id"):
        mock_get_gym_id = mocker.patch("hud.gym.get_gym_id", new_callable=AsyncMock)
        mock_get_gym_id.return_value = "true-gym-id"

    # Mock the _setup method to avoid the config requirement
    mocker.patch("hud.env.environment.Environment._setup", new_callable=AsyncMock)

    env = await make(test_case["env_src"])

    assert isinstance(env, Environment)
    assert env.client == mock_client
    assert env.build_data == mock_build_data
    if isinstance(test_case["expected_create_args"], tuple):
        mock_create.assert_called_once_with(*test_case["expected_create_args"])
    else:
        mock_create.assert_called_once_with(**test_case["expected_create_args"])
    if "expected_source_path" in test_case:
        mock_client.set_source_path.assert_called_once_with(Path(test_case["expected_source_path"]))


@pytest.mark.asyncio
async def test_make_with_job_association(mocker):
    """Test creating an environment with job association."""
    mock_get_gym_id = mocker.patch("hud.gym.get_gym_id", new_callable=AsyncMock)
    mock_get_gym_id.return_value = "true-gym-id"

    mock_client = MockClient()
    mock_build_data = {"image": "test-image"}
    mock_create = mocker.patch("hud.gym.RemoteClient.create", new_callable=AsyncMock)
    mock_create.return_value = (mock_client, mock_build_data)

    job = Job(
        id="test-job-123",
        name="Test Job",
        metadata={"test": "data"},
        created_at=datetime.datetime.now(),
        status="created",
    )

    # Mock the _setup method to avoid the config requirement
    mocker.patch("hud.env.environment.Environment._setup", new_callable=AsyncMock)

    env = await make("qa", job=job)
    assert isinstance(env, Environment)
    assert env.client == mock_client
    assert env.build_data == mock_build_data
    mock_create.assert_called_once_with(
        gym_id="true-gym-id", job_id=job.id, task_id=None, metadata={}
    )


@pytest.mark.asyncio
async def test_make_with_invalid_gym():
    """Test creating an environment with an invalid gym source."""
    with pytest.raises(ValueError, match="Invalid gym source"):
        # Create a mock object that is neither a Gym nor a Task
        mock_invalid = MagicMock()
        mock_invalid.__class__ = type("InvalidGym", (), {})
        await make(mock_invalid)


@pytest.mark.asyncio
async def test_make_with_invalid_location():
    """Test creating an environment with an invalid location."""
    # Create a CustomGym instance with an invalid location
    with pytest.raises(ValueError, match="Invalid environment location"):
        await make(
            MagicMock(
                spec=CustomGym,
                dockerfile="test.Dockerfile",
                location="invalid",
                controller_source_dir="/path/to/source",
            )
        )


@pytest.mark.asyncio
async def test_make_without_dockerfile():
    """Test creating an environment without a dockerfile."""
    # Create a CustomGym instance without a dockerfile
    with pytest.raises(ValueError, match="Dockerfile is required for custom environments"):
        await make(
            MagicMock(
                spec=CustomGym,
                dockerfile=None,
                location="local",
                controller_source_dir="/path/to/source",
            )
        )
