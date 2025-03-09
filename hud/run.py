from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .adapters.common import Adapter
from .environment import Environment, EvalSet
from .server import make_request
from .settings import settings

if TYPE_CHECKING:
    import datetime

    from .gym import Gym


class RunResponse(BaseModel):
    """
    Response model for run data from the API.

    Attributes:
        id: Unique identifier for the run
        name: Human-readable name of the run
        gym: Dictionary containing gym information
        evalset: Dictionary containing evalset information
        config: Dictionary containing configuration parameters
        metadata: Dictionary containing metadata
    """

    id: str
    name: str
    gym: dict[str, Any]
    evalset: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]


class RunAnalyticsResponse(BaseModel):
    """
    Model for Run analytics data.

    Attributes:
        id: Unique identifier for the run
        name: Human-readable name of the run
        status_counts: Counts of tasks in different states
        avg_score: Average score across all tasks, if available
        completion_rate: Percentage of tasks completed
        total_tasks: Total number of tasks in the run
        completed_tasks: Number of completed tasks
        running_time: Total runtime in seconds, if available
        created_at: When the run was created
        raw_data: Detailed data about tasks and environments
    """

    id: str
    name: str
    status_counts: dict[str, int]  # e.g. {"completed": 5, "running": 2, "error": 1}
    avg_score: float | None = None
    completion_rate: float | None = None  # percentage of tasks completed
    total_tasks: int
    completed_tasks: int
    running_time: float | None = None  # runtime in seconds if available
    created_at: datetime.datetime
    raw_data: dict[str, list[dict[str, Any]]] = Field(
        default_factory=lambda: {"tasks": [], "environments": []}
    )

    def __str__(self) -> str:
        return self.visualize()

    def visualize(self) -> str:
        """
        Generate an ASCII bar chart visualization of run analytics.

        Args:
            data: The run analytics data to visualize

        Returns:
            A string containing an ASCII visualization
        """
        max_width = 50

        completion_rate = self.completion_rate if self.completion_rate is not None else 0

        result = [
            f"Run: {self.name} (ID: {self.id})",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 60,
            f"""Progress: {self.completed_tasks}/{self.total_tasks} tasks completed (
            {completion_rate:.1f}% completion rate)""",
            "",
        ]

        result.append("Status Distribution:")
        total = sum(self.status_counts.values())
        for status, count in self.status_counts.items():
            percentage = (count / total) * 100
            bar_length = int((count / total) * max_width)
            bar = "█" * bar_length
            result.append(f"{status.ljust(10)}: {bar} {count} ({percentage:.1f}%)")

        if self.avg_score is not None:
            result.append("")
            result.append(f"Average Score: {self.avg_score:.2f}")

            score_bar_length = int((self.avg_score / 100) * max_width)
            score_bar = "█" * score_bar_length
            result.append(f"Score: {score_bar} {self.avg_score:.2f}/1.00")

        if self.running_time is not None:
            hours, remainder = divmod(self.running_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            result.append(f"Total Runtime: {runtime_str}")

        return "\n".join(result)


class Run:
    """
    A run represents a collection of tasks and environments.

    This class provides methods to fetch task IDs, create environments,
    and access analytics for the run.
    """

    def __init__(
        self,
        id: str,
        name: str,
        gym: Gym,
        evalset: EvalSet,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        adapter: Adapter | None = None,
    ) -> None:
        """
        Initialize a run.

        Args:
            id: Unique identifier
            name: Human-readable name
            gym: Gym object for this run
            evalset: EvalSet object containing tasks
            config: Optional configuration parameters
            metadata: Optional metadata
            adapter: Optional adapter for action conversion
        """
        adapter = adapter or Adapter()
        if metadata is None:
            metadata = {}
        if config is None:
            config = {}
        self.id = id
        self.name = name
        self.gym = gym
        self.evalset = evalset
        self.adapter = adapter
        self.config = config
        self.metadata = metadata
        self.environments: list[Environment] = []

    async def fetch_task_ids(self) -> list[str]:
        """
        Fetch task IDs for this run from the evalset.

        Returns:
            list[str]: List of task IDs
        """
        return await self.evalset.fetch_tasks()

    async def make(self, metadata: dict[str, Any] | None = None) -> Environment:
        """
        Create a new environment for this run.

        Args:
            metadata: Metadata for the environment

        Returns:
            Environment: The created environment
        """
        # Make the env class
        env = Environment(
            run_id=self.id,
            config=self.config,
            adapter=self.adapter,
            metadata=metadata or {},
        )
        await env.create_environment()
        self.environments.append(env)
        return env

    async def get_analytics(self) -> RunAnalyticsResponse:
        """
        Get analytics for this run.

        Returns:
            RunAnalyticsResponse: Analytics data including status counts,
                                average score, and other metrics
        """
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/runs/{self.id}/analytics",
            api_key=settings.api_key,
        )
        return RunAnalyticsResponse(**data)
