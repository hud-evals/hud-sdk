from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter

from hud.server import make_request
from hud.settings import settings
from hud.task import Task

if TYPE_CHECKING:
    from inspect_ai.dataset import Dataset


async def fetch_task_ids(taskset_id: str, *, api_key: str | None = None) -> list[str]:
    """
    Fetch all task IDs in this taskset from the API.
    
    Returns:
        list[str]: List of task IDs
    """
    
    if api_key is None:
        api_key = settings.api_key
    
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/evalsets/{taskset_id}/tasks",
        api_key=api_key,
    )
    return [task_id for task_id in data["tasks"]]
    

class TaskSet(BaseModel):
    """
    Collection of related tasks for benchmarking.

    Attributes:
        id: Unique identifier for the taskset
        description: Description of the taskset
        tasks: List of Task objects in the taskset
    """
    id: str | None = None
    description: str | None = None
    tasks: list[Task] = []
    
    @classmethod
    async def load(cls, taskset_id: str, api_key: str | None = None) -> TaskSet:
        """
        Loads a TaskSet by its ID.

        Args:
            taskset_id: The ID of the taskset to load
            api_key: Optional API key to use for the request

        Returns:
            TaskSet: The loaded taskset
        """
        
        if api_key is None:
            api_key = settings.api_key
        
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/evalsets/{taskset_id}/tasks",
            api_key=api_key,
        )
        tasks = TypeAdapter(list[Task]).validate_python(data["evalset"])
        
        return cls(
            id=taskset_id,
            tasks=tasks,
        )
    
    @classmethod
    def from_inspect_dataset(cls, dataset: Dataset) -> TaskSet:
        """
        Creates a TaskSet from an inspect-ai dataset.

        Args:
            dataset: An inspect-ai dataset

        Returns:
            TaskSet: A new TaskSet instance
        """
        tasks = [Task.from_inspect_sample(sample) for sample in dataset ]
    
        return cls(
            id=None,
            tasks=tasks,
            description=None,
        )
