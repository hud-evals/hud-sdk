from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from .adapters.common import Adapter
from .environment import Environment
from .evalset import EvalSet
from .server import make_request
from .settings import settings

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


async def load_run(id: str, *, api_key: Optional[str]=None) -> Optional[Run]:
    """
    Load a run by ID from the HUD API.

    Args:
        id: The ID of the run to load
        adapter: Optional adapter for action conversion

    Returns:
        Run: The loaded run object, or None if not found
    """
    if api_key is None:
        api_key = settings.api_key
    
    
    # API call to get run info
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/runs/{id}",
        api_key=api_key,
    )
    if data:
        response = RunResponse(**data)
        evalset = EvalSet(
            id=response.evalset["id"],
            name=response.evalset["name"],
            tasks=response.evalset["tasks"],
        )
        return Run(
            id=response.id,
            name=response.name,
            metadata=response.metadata,
        )
    return None

async def make_run(
    name: str,
    gym_id: str,
    evalset_id: str,
    *,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    api_key: str | None = None,
):
    """
    Create a new run in the HUD system.

    Args:
        name: Name of the run
        gym: Gym to use for the run
        evalset: Evalset to use for the run
        config: Optional configuration parameters
        metadata: Optional metadata for the run
        adapter: Optional adapter for action conversion

    Returns:
        Run: The created run object
    """
    if api_key is None:
        api_key = settings.api_key

    data = await make_request(
        method="POST",
        url=f"{settings.base_url}/runs",
        json={
            "name": name,
            "gym_id": gym_id,
            "evalset_id": evalset_id,
            "config": json.dumps(config),
            "metadata": json.dumps(metadata),
        },
        api_key=api_key,
    )
    
    # TODO: determine which fields are necessary here
    return Run(
        id=data["id"],
        name=data["name"],
        metadata=data["metadata"],
    )

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
        metadata: dict[str, Any] | None = None,
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
        self.id = id
        self.name = name
        self.metadata = metadata

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

