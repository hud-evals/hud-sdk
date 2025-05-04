from __future__ import annotations

import datetime
import datetime as dt
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hud.job import Job, create_job, job, load_job, run_job


@pytest.fixture
def mock_job_data() -> dict[str, Any]:
    return {
        "id": "test-job-123",
        "name": "Test Job",
        "metadata": {"test": "data"},
        "created_at": datetime.datetime.now().isoformat(),
        "status": "created",
    }


@pytest.mark.asyncio
async def test_create_job(mock_job_data: dict[str, Any], mocker) -> None:
    """Test that a job can be created as expected from the response data."""
    mock_make_request = mocker.patch("hud.job.make_request", new_callable=AsyncMock)
    mock_make_request.return_value = mock_job_data
    result = await create_job(name="Test Job", metadata={"test": "data"})
    assert isinstance(result, Job)
    assert result.id == mock_job_data["id"]
    assert result.name == mock_job_data["name"]
    assert result.metadata == mock_job_data["metadata"]
    assert result.status == mock_job_data["status"]
    assert result.created_at == dt.datetime.fromisoformat(mock_job_data["created_at"])


@pytest.mark.asyncio
async def test_load_job(mock_job_data: dict[str, Any], mocker) -> None:
    """Test that a job can be loaded as expected from the response data."""
    mock_make_request = mocker.patch("hud.job.make_request", new_callable=AsyncMock)
    mock_make_request.return_value = mock_job_data
    result = await load_job(job_id="test-job-123")
    assert isinstance(result, Job)
    assert result.id == mock_job_data["id"]
    assert result.name == mock_job_data["name"]
    assert result.metadata == mock_job_data["metadata"]
    assert result.status == mock_job_data["status"]
    assert result.created_at == dt.datetime.fromisoformat(mock_job_data["created_at"])


@pytest.mark.asyncio
async def test_job_decorator(mocker):
    """Test that the job decorator works as expected."""

    @job(name="Decorated Job", metadata={"test": "decorator"})
    async def test_function():
        return "test result"

    mock_create_job = mocker.patch("hud.job.create_job", new_callable=AsyncMock)
    mock_create_job.return_value = Job(
        id="decorated-job-123",
        name="Decorated Job",
        metadata={"test": "decorator"},
        created_at=datetime.datetime.now(),
        status="created",
    )
    await test_function()
    mock_create_job.assert_called_once_with(name="Decorated Job", metadata={"test": "decorator"})


@pytest.mark.asyncio
async def test_run_job(mocker):
    """Test that run_job works as expected with both single task and taskset."""
    from hud.adapters.common import Adapter
    from hud.agent.base import Agent
    from hud.task import Task
    from hud.taskset import TaskSet

    class MockAgent(Agent):
        async def predict(self, obs):
            return "action", True

        async def fetch_response(self, prompt: str) -> str:
            return "mock response"

    class MockAdapter(Adapter):
        pass

    mock_task = Task(id="test-task-1", prompt="Test Task")
    mock_taskset = TaskSet(tasks=[mock_task])

    mock_create_job = mocker.patch("hud.job.create_job", new_callable=AsyncMock)
    mock_create_job.return_value = Job(
        id="test-job-123",
        name="Test Job",
        metadata={"test": "data"},
        created_at=datetime.datetime.now(),
        status="created",
    )

    mock_gym_make = mocker.patch("hud.gym.make", new_callable=AsyncMock)
    mock_env = AsyncMock()
    mock_env.reset.return_value = ("obs", {})
    mock_env.step.return_value = ("obs", 0, True, {})
    mock_env.evaluate.return_value = {"success": True}
    mock_gym_make.return_value = mock_env

    job = await run_job(
        agent_cls=MockAgent,
        task_or_taskset=mock_task,
        job_name="Test Job",
        adapter_cls=MockAdapter,
        max_steps_per_task=5,
        run_parallel=False,
        show_progress=False,
    )

    assert job.id == "test-job-123"
    assert job.name == "Test Job"
    assert job.metadata == {"test": "data"}
    mock_create_job.assert_called_once_with(
        name="Test Job",
        metadata=None,
    )
    mock_gym_make.assert_called_once()

    mock_create_job.reset_mock()
    mock_gym_make.reset_mock()

    job = await run_job(
        agent_cls=MockAgent,
        task_or_taskset=mock_taskset,
        job_name="Test Job",
        adapter_cls=MockAdapter,
        max_steps_per_task=5,
        run_parallel=True,
        show_progress=False,
    )

    assert job.id == "test-job-123"
    assert job.name == "Test Job"
    assert job.metadata == {"test": "data"}
    mock_create_job.assert_called_once_with(
        name="Test Job",
        metadata=None,
    )
    mock_gym_make.assert_called_once()


@pytest.mark.asyncio
async def test_get_active_job(mocker):
    """Test that get_active_job correctly retrieves the active job from the call stack."""
    from hud.job import get_active_job

    assert get_active_job() is None

    mock_job = Job(
        id="active-job-123",
        name="Active Job",
        metadata={"test": "data"},
        created_at=datetime.datetime.now(),
        status="created",
    )

    @job(name="Test Job", metadata={"test": "decorator"})
    async def test_function():
        """Test that while in the scope of a job, get_active_job returns the active job."""
        active_job = get_active_job()
        assert active_job is not None
        assert active_job.id == mock_job.id
        assert active_job.name == mock_job.name
        assert active_job.metadata == mock_job.metadata

    mock_create_job = mocker.patch("hud.job.create_job", new_callable=AsyncMock)
    mock_create_job.return_value = mock_job

    await test_function()
    # Should go back to None after the function returns.
    assert get_active_job() is None
