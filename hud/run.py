from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .adapters.common import Adapter
from .env import Env, EvalSet
from .server import make_request
from .settings import settings

if TYPE_CHECKING:
    from datetime import datetime

    from .gym import Gym


class RunResponse(BaseModel):
    """
    Response model for run data from the API.
    
    Attributes:
        id: Unique identifier for the run
        name: Human-readable name of the run
        gym: Dictionary containing gym information
        evalset: Dictionary containing evalset information
        adapter: Dictionary containing adapter information
        config: Dictionary containing configuration parameters
        metadata: Dictionary containing metadata
    """
    id: str
    name: str
    gym: dict[str, Any]
    evalset: dict[str, Any]
    adapter: dict[str, Any]
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
    created_at: datetime
    raw_data: dict[str, list[dict[str, Any]]] = Field(
        default_factory=lambda: {"tasks": [], "environments": []}
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
        self.envs: list[Env] = []

    async def fetch_task_ids(self) -> list[str]:
        """
        Fetch task IDs for this run from the evalset.
        
        Returns:
            list[str]: List of task IDs
        """
        return await self.evalset.fetch_tasks()

    async def make(self, metadata: dict[str, Any]) -> Env:
        """
        Create a new environment for this run.
        
        Args:
            metadata: Metadata for the environment
            
        Returns:
            Env: The created environment
        """
        # Make the env class
        env = Env(
            run_id=self.id,
            config=self.config,
            adapter=self.adapter,
            metadata=metadata,
        )
        await env.create_environment()
        self.envs.append(env)
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
