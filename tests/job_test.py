from __future__ import annotations

import datetime
from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.adapters.common import Adapter
from hud.agent.base import Agent
from hud.job import Job, create_job, get_active_job, job, load_job, run_job
from hud.settings import settings
from hud.task import Task
from hud.taskset import TaskSet


class MockAgent(Agent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.predict = AsyncMock(return_value=(None, True))
        self.reset = AsyncMock()


@pytest.fixture
def mock_job_data():
    return {
        "id": "test-job-123",
        "name": "Test Job",
        "metadata": {"test": "data"},
        "created_at": datetime.datetime.now().isoformat(),
        "status": "created",
    }


@pytest.fixture
def mock_task():
    task = MagicMock(spec=Task)
    task.id = "test-task-123"
    return task


@pytest.fixture
def mock_taskset():
    taskset = MagicMock(spec=TaskSet)
    taskset.tasks = [MagicMock(spec=Task, id=f"task-{i}") for i in range(3)]
    return taskset


@pytest.mark.asyncio
async def test_create_job(mock_job_data):
    with patch("hud.job.make_request", new_callable=AsyncMock) as mock_make_request:
        mock_make_request.return_value = mock_job_data

        result = await create_job(name="Test Job", metadata={"test": "data"})

        assert isinstance(result, Job)
        assert result.id == mock_job_data["id"]
        assert result.name == mock_job_data["name"]
        assert result.metadata == mock_job_data["metadata"]
        assert result.status == mock_job_data["status"]


# @pytest.mark.asyncio
# async def test_load_job(mock_job_data):
#     with patch("hud.job.make_request", new_callable=AsyncMock) as mock_make_request:
#         mock_make_request.return_value = mock_job_data

#         result = await load_job("test-job-123")

#         assert isinstance(result, Job)
#         assert result.id == mock_job_data["id"]
#         assert result.name == mock_job_data["name"]


# @pytest.mark.asyncio
# async def test_job_decorator():
#     @job(name="Decorated Job", metadata={"test": "decorator"})
#     async def test_function():
#         return "test result"

#     with patch("hud.job.create_job", new_callable=AsyncMock) as mock_create_job:
#         mock_create_job.return_value = Job(
#             id="decorated-job-123",
#             name="Decorated Job",
#             metadata={"test": "decorator"},
#             created_at=datetime.datetime.now(),
#             status="created",
#         )

#         result = await test_function()

#         assert result == "test result"
#         mock_create_job.assert_called_once_with(
#             name="Decorated Job", metadata={"test": "decorator"}
#         )


# @pytest.mark.asyncio
# async def test_get_active_job():
#     # Test when no active job exists
#     assert get_active_job() is None

#     # Test with active job
#     test_job = Job(
#         id="active-job-123",
#         name="Active Job",
#         metadata={},
#         created_at=datetime.datetime.now(),
#         status="created",
#     )

#     with patch("hud.job._ACTIVE_JOBS", {"test_function_123": test_job}):
#         with patch("hud.job.inspect.currentframe") as mock_frame:
#             mock_frame.return_value.f_locals = {"_job_call_id": "test_function_123"}
#             assert get_active_job() == test_job


# @pytest.mark.asyncio
# async def test_run_job_single_task(mock_task):
#     mock_agent_cls = MockAgent

#     with patch("hud.job.create_job", new_callable=AsyncMock) as mock_create_job:
#         mock_create_job.return_value = Job(
#             id="run-job-123",
#             name="Run Job Test",
#             metadata={},
#             created_at=datetime.datetime.now(),
#             status="created",
#         )

#         with patch("hud.job.gym.make", new_callable=AsyncMock) as mock_gym_make:
#             mock_env = MagicMock()
#             mock_env.reset = AsyncMock(return_value=(None, None))
#             mock_env.step = AsyncMock(return_value=(None, None, True, None))
#             mock_env.evaluate = AsyncMock(return_value={"success": True})
#             mock_env.close = AsyncMock()
#             mock_gym_make.return_value = mock_env

#             result = await run_job(
#                 agent_cls=mock_agent_cls,
#                 task_or_taskset=mock_task,
#                 job_name="Run Job Test",
#                 run_parallel=False,
#             )

#             assert isinstance(result, Job)
#             assert result.id == "run-job-123"
#             mock_gym_make.assert_called_once()


# @pytest.mark.asyncio
# async def test_run_job_taskset(mock_taskset):
#     mock_agent_cls = MockAgent

#     with patch("hud.job.create_job", new_callable=AsyncMock) as mock_create_job:
#         mock_create_job.return_value = Job(
#             id="run-job-123",
#             name="Run Job Test",
#             metadata={},
#             created_at=datetime.datetime.now(),
#             status="created",
#         )

#         with patch("hud.job.gym.make", new_callable=AsyncMock) as mock_gym_make:
#             mock_env = MagicMock()
#             mock_env.reset = AsyncMock(return_value=(None, None))
#             mock_env.step = AsyncMock(return_value=(None, None, True, None))
#             mock_env.evaluate = AsyncMock(return_value={"success": True})
#             mock_env.close = AsyncMock()
#             mock_gym_make.return_value = mock_env

#             result = await run_job(
#                 agent_cls=mock_agent_cls,
#                 task_or_taskset=mock_taskset,
#                 job_name="Run Job Test",
#                 run_parallel=True,
#                 max_concurrent_tasks=2,
#             )

#             assert isinstance(result, Job)
#             assert result.id == "run-job-123"
#             assert mock_gym_make.call_count == len(mock_taskset.tasks)


# @pytest.mark.asyncio
# async def test_job_load_trajectories():
#     job = Job(
#         id="test-job-123",
#         name="Test Job",
#         metadata={},
#         created_at=datetime.datetime.now(),
#         status="created",
#     )

#     mock_trajectories = [{"id": f"traj-{i}"} for i in range(3)]

#     with patch("hud.job.make_request", new_callable=AsyncMock) as mock_make_request:
#         mock_make_request.return_value = mock_trajectories

#         trajectories = await job.load_trajectories()

#         assert len(trajectories) == len(mock_trajectories)
#         mock_make_request.assert_called_once_with(
#             method="GET",
#             url=f"{settings.base_url}/v2/jobs/{job.id}/trajectories",
#             api_key=settings.api_key,
#         )


# @pytest.mark.asyncio
# async def test_job_get_analytics():
#     job = Job(
#         id="test-job-123",
#         name="Test Job",
#         metadata={},
#         created_at=datetime.datetime.now(),
#         status="created",
#     )

#     mock_trajectories = [MagicMock(reward=1.0), MagicMock(reward=0.5), MagicMock(reward=1.0)]

#     with patch.object(job, "load_trajectories", new_callable=AsyncMock) as mock_load_trajectories:
#         mock_load_trajectories.return_value = mock_trajectories

#         analytics = await job.get_analytics()

#         assert "task_count" in analytics
#         assert "avg_reward" in analytics
#         assert "success_rate" in analytics
#         assert analytics["task_count"] == len(mock_trajectories)
#         assert analytics["avg_reward"] == (1.0 + 0.5 + 1.0) / 3
