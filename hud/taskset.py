from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from hud.server import make_request
from hud.settings import settings
from hud.task import Task

if TYPE_CHECKING:
    from inspect_ai.dataset import Dataset


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
        
        return cls.model_validate({
            "id": taskset_id,
            "tasks": data["evalset"],
        })
    
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
