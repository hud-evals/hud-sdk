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
from hud.gym import CustomGym, make
from hud.job import Job
from hud.task import Task
from hud.telemetry import reset_context
from hud.types import EnvironmentStatus
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
            "client_class": "hud.gym.LocalDockerClient",
            "expected_create_args": {
                "image": "test-image:latest",
                "host_config": {"NetworkMode": "host"},
                "remote_logging_for_local_docker": False,
            },
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
            "expected_create_args": {
                "image": "test-image:latest",
                "remote_logging_for_local_docker": False,
            },
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

    if isinstance(test_case.get("expected_create_args"), dict):
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
        # Create an object that is neither a Gym nor a Task
        class InvalidGym:
            pass

        invalid_gym = InvalidGym()
        await make(invalid_gym)  # type: ignore[arg-type]


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        # Test that local CustomGym defaults to autolog=False
        {
            "name": "local_gym_default_autolog",
            "env_src": CustomGym(
                location="local",
                image_or_build_context="test-image:latest",
            ),
            "expected_autolog": False,
            "expected_create_args": {
                "image": "test-image:latest",
                "remote_logging_for_local_docker": False,
            },
        },
        # Test that remote CustomGym defaults to autolog=True
        {
            "name": "remote_gym_default_autolog",
            "env_src": CustomGym(
                location="remote",
                image_or_build_context="test-image:latest",
            ),
            "expected_autolog": True,
            "expected_create_args": {
                "image_uri": "test-image:latest",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
        },
        # Test that we can override local CustomGym to use autolog=True
        {
            "name": "local_gym_explicit_autolog",
            "env_src": CustomGym(
                location="local",
                image_or_build_context="test-image:latest",
            ),
            "autolog": True,
            "expected_autolog": True,
            "expected_create_args": {
                "image": "test-image:latest",
                "remote_logging_for_local_docker": True,
            },
        },
        # Test that we can override remote CustomGym to use autolog=False
        {
            "name": "remote_gym_explicit_autolog",
            "env_src": CustomGym(
                location="remote",
                image_or_build_context="test-image:latest",
            ),
            "autolog": False,
            "expected_autolog": False,
            "expected_create_args": {
                "image_uri": "test-image:latest",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
        },
        # Test that the 'qa' ServerGym defaults to autolog=True
        {
            "name": "qa_gym_default_autolog",
            "env_src": "qa",
            "expected_autolog": True,
            "expected_create_args": {
                "gym_id": "qa",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
        },
        # Test that we can override 'qa' ServerGym to use autolog=False
        {
            "name": "qa_gym_explicit_autolog_false",
            "env_src": "qa",
            "autolog": False,
            "expected_autolog": False,
            "expected_create_args": {
                "gym_id": "qa",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
        },
        # Test that the 'hud-browser' ServerGym defaults to autolog=True
        {
            "name": "hud_browser_default_autolog",
            "env_src": "hud-browser",
            "expected_autolog": True,
            "expected_create_args": {
                "gym_id": "hud-browser",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
        },
        # Test that the 'OSWorld-Ubuntu' ServerGym defaults to autolog=True
        {
            "name": "osworld_ubuntu_default_autolog",
            "env_src": "OSWorld-Ubuntu",
            "expected_autolog": True,
            "expected_create_args": {
                "gym_id": "OSWorld-Ubuntu",
                "job_id": None,
                "task_id": None,
                "metadata": {},
            },
        },
    ],
)
async def test_make_autolog_behavior(mocker, test_case):
    """Test the autolog behavior for different gym types and configurations."""
    reset_context()
    mock_client = MockClient()
    mock_build_data = {"image": "test-image"}

    # Mock RemoteClient.create for string-based gyms
    mock_remote_create = mocker.patch("hud.gym.RemoteClient.create", new_callable=AsyncMock)
    mock_remote_create.return_value = (mock_client, mock_build_data)

    # Mock LocalDockerClient.create for local gyms
    mock_local_create = mocker.patch("hud.gym.LocalDockerClient.create", new_callable=AsyncMock)
    mock_local_create.return_value = mock_client

    # Mock RemoteDockerClient.create for remote gyms
    mock_remote_docker_create = mocker.patch("hud.gym.RemoteDockerClient.create", new_callable=AsyncMock)
    mock_remote_docker_create.return_value = mock_client

    # Mock get_gym_id for string-based gyms
    mock_get_gym_id = mocker.patch("hud.gym.get_gym_id", new_callable=AsyncMock)
    mock_get_gym_id.return_value = test_case["env_src"] if isinstance(test_case["env_src"], str) else None

    # Mock the _setup method to avoid the config requirement
    mocker.patch("hud.env.environment.Environment._setup", new_callable=AsyncMock)

    # Create environment with optional autolog parameter
    kwargs = {"autolog": test_case["autolog"]} if "autolog" in test_case else {}
    env = await make(test_case["env_src"], **kwargs)

    assert isinstance(env, Environment)
    assert env.autolog == test_case["expected_autolog"]

    # Verify the correct client creation was called with expected args
    if isinstance(test_case["env_src"], str):
        mock_remote_create.assert_called_once_with(**test_case["expected_create_args"])
    elif isinstance(test_case["env_src"], CustomGym):
        if test_case["env_src"].location == "local":
            mock_local_create.assert_called_once_with(**test_case["expected_create_args"])
        else:
            mock_remote_docker_create.assert_called_once_with(**test_case["expected_create_args"])
