from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ConfigDict

from hud.env.client import Client
from hud.env.environment import Environment
from hud.exceptions import GymMakeException
from hud.gym import make
from hud.job import Job
from hud.task import Task
from hud.telemetry.context import reset_context
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
                location="local",
                image_or_build_context=Path("/path/to/source"),
            ),
            "client_class": "hud.gym.LocalDockerClient",
            "expected_build_args": (Path("/path/to/source"),),
            "expected_source_path": Path("/path/to/source"),
            "config": [FunctionConfig(function="test", args=[])],
            "check_build_data": False,
        },
        {
            "name": "custom_local_gym_with_host_config",
            "env_src": CustomGym(
                location="local",
                image_or_build_context="test-image:latest",
                host_config={"NetworkMode": "host"},
            ),
            "expected_create_args": ("test-image:latest",{"NetworkMode": "host"}),
            "config": [FunctionConfig(function="test", args=[])],
            "check_build_data": False,
        },
        {
            "name": "custom_local_gym_with_image",
            "env_src": CustomGym(
                location="local",
                image_or_build_context="test-image:latest",
            ),
            "client_class": "hud.gym.LocalDockerClient",
            "expected_create_args": ("test-image:latest",),
            "config": [FunctionConfig(function="test", args=[])],
            "check_build_data": False,
        },
        {
            "name": "custom_remote_gym",
            "env_src": Task(
                id="test-task-1",
                prompt="Test Task",
                gym=CustomGym(
                    location="remote",
                    image_or_build_context=Path("/path/to/source"),
                ),
            ),
            "client_class": "hud.gym.RemoteDockerClient",
            "expected_create_args": {
                "image_uri": "test-image",
                "job_id": None,
                "task_id": "test-task-1",
                "metadata": {},
            },
            "check_build_data": True,
        },
    ],
)
async def test_make_docker_gym(mocker, test_case):
    """Test creating environments with different gym types."""
    reset_context()
    mock_client = MockClient()
    mock_build_data = {"image": "test-image"}

    mock_build_image = mocker.patch(
        f"{test_case['client_class']}.build_image", new_callable=AsyncMock
    )
    mock_build_image.return_value = (mock_build_data["image"], mock_build_data)

    mock_create = mocker.patch(f"{test_case['client_class']}.create", new_callable=AsyncMock)
    mock_create.return_value = mock_client

    if test_case.get("mock_get_gym_id"):
        mock_get_gym_id = mocker.patch("hud.gym.get_gym_id", new_callable=AsyncMock)
        mock_get_gym_id.return_value = "true-gym-id"

    # Mock the _setup method to avoid the config requirement
    mocker.patch("hud.env.environment.Environment._setup", new_callable=AsyncMock)

    env = await make(test_case["env_src"])

    assert isinstance(env, Environment)
    assert env.client == mock_client
    if test_case.get("check_build_data"):
        assert env.build_data == mock_build_data

    if isinstance(test_case.get("expected_build_args"), tuple):
        mock_build_image.assert_called_once_with(*test_case["expected_build_args"])
    elif isinstance(test_case.get("expected_build_args"), dict):
        mock_build_image.assert_called_once_with(**test_case["expected_build_args"])

    if isinstance(test_case.get("expected_create_args"), tuple):
        mock_create.assert_called_once_with(*test_case["expected_create_args"])
    elif isinstance(test_case.get("expected_create_args"), dict):
        mock_create.assert_called_once_with(**test_case["expected_create_args"])

    if "expected_source_path" in test_case:
        mock_client.set_source_path.assert_called_once_with(test_case["expected_source_path"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    (
        {
            "env_src": "qa",
            "expected_create_args": {
                "gym_id": "qa",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
        },
        {
            "env_src": "novnc_ubuntu",
            "expected_create_args": {
                "gym_id": "novnc_ubuntu",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
        },
    ),
)
async def test_make_remote_gym(mocker, test_case):
    reset_context()
    mock_client = MockClient()
    mock_build_data = {"image": "test-image"}

    mock_create = mocker.patch("hud.gym.RemoteClient.create", new_callable=AsyncMock)
    mock_create.return_value = mock_client, mock_build_data

    mock_get_gym_id = mocker.patch("hud.gym.get_gym_id", new_callable=AsyncMock)
    mock_get_gym_id.return_value = test_case["env_src"]

    # Mock the _setup method to avoid the config requirement
    mocker.patch("hud.env.environment.Environment._setup", new_callable=AsyncMock)

    env = await make(test_case["env_src"])

    assert isinstance(env, Environment)
    assert env.client == mock_client
    assert env.build_data == mock_build_data
    mock_create.assert_called_once_with(**test_case["expected_create_args"])


@pytest.mark.asyncio
async def test_make_with_job_association(mocker):
    """Test creating an environment with job association."""
    reset_context()
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
    reset_context()
    with pytest.raises(GymMakeException, match="Invalid gym source"):
        # Create a mock object that is neither a Gym nor a Task
        mock_invalid = MagicMock()
        mock_invalid.__class__ = type("InvalidGym", (), {})
        await make(mock_invalid)


@pytest.mark.asyncio
async def test_make_with_invalid_location():
    """Test creating an environment with an invalid location."""
    reset_context()
    # Create a CustomGym instance with an invalid location
    with pytest.raises(GymMakeException, match="Invalid environment location"):
        await make(
            MagicMock(
                spec=CustomGym,
                location="invalid",
                image_or_build_context=Path("/path/to/source"),
            )
        )


@pytest.mark.asyncio
async def test_make_without_image_or_build_context():
    """Test creating an environment without an image or build context."""
    reset_context()
    # Create a CustomGym instance without an image or build context
    with pytest.raises(GymMakeException, match="Invalid image or build context"):
        await make(MagicMock(spec=CustomGym, location="local", image_or_build_context=None))
