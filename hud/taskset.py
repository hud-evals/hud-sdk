from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from hud.server import make_request
from hud.settings import settings
from hud.task import Task

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    
    def __getitem__(self, index: int) -> Task:
        """
        Allows accessing tasks by index using square bracket notation.

        Args:
            index: The index of the task to retrieve

        Returns:
            Task: The task at the specified index

        Raises:
            IndexError: If the index is out of range
        """
        return self.tasks[index]
    
    def __len__(self) -> int:
        """
        Returns the number of tasks in the taskset.

        Returns:
            int: The number of tasks in the taskset
        """
        return len(self.tasks)
    
    def __iter__(self) -> Iterator[Task]:
        """
        Returns an iterator over the tasks in the taskset.
        """
        return iter(self.tasks)

    
async def load_taskset(taskset_id: str, api_key: str | None = None) -> TaskSet:
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
        url=f"{settings.base_url}/v2/tasksets/{taskset_id}/tasks",
        api_key=api_key,
    )
    
    return TaskSet.model_validate({
        "id": taskset_id,
        "tasks": data["evalset"],
    })

def load_from_inspect(dataset: Dataset) -> TaskSet:
    """
    Creates a TaskSet from an inspect-ai dataset.

    Args:
        dataset: An inspect-ai dataset

    Returns:
        TaskSet: A new TaskSet instance
    """
    tasks = [Task.from_inspect_sample(sample) for sample in dataset]

    return TaskSet(
        id=None,
        tasks=tasks,
        description=dataset.name,
    )
