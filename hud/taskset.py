from __future__ import annotations

from typing import Optional, Any
from hud.settings import settings
from hud.server import make_request
from hud.task import Task


async def fetch_task_ids(taskset_id: str, *, api_key: Optional[str] = None) -> list[str]:
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
    

class TaskSet:
    """
    Collection of related tasks for benchmarking.

    Attributes:
        id: Unique identifier for the taskset
        description: Description of the taskset
        tasks: List of Task objects in the taskset
    """

    def __init__(
        self,
        id: str,
        tasks: list[Task],
        description: str = "",
    ) -> None:
        """
        Initialize a task set.

        Args:
            id: Unique identifier
            tasks: List of Task objects
            description: Optional description of the taskset
        """
        self.id = id
        self.tasks = tasks
        self.description = description

    @classmethod
    async def load(cls, taskset_id: str, api_key: Optional[str] = None) -> TaskSet:
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
        tasks = []
        if "evalset" in data:
            for task in data["evalset"]:
                tasks.append(Task(**task))
        else:
            for task in data["tasks"]:
                tasks.append(Task(id=task))
        
        return cls(
            id=taskset_id,
            tasks=tasks,
        )
    
    @classmethod
    def from_inspect_dataset(cls, dataset: Any) -> TaskSet:
        """
        Creates a TaskSet from an inspect-ai dataset.

        Args:
            dataset: An inspect-ai dataset

        Returns:
            TaskSet: A new TaskSet instance
        """
        # Implementation would go here
        raise NotImplementedError("from_inspect_dataset is not yet implemented")

    def __getitem__(self, index: int) -> Task:
        """
        Returns the task at the specified index.
        
        Args:
            index: The index of the task to retrieve
            
        Returns:
            Task: The task at the specified index
        """
        return self.tasks[index]
        
    def __len__(self) -> int:
        """
        Returns the number of tasks in the taskset.
        
        Returns:
            int: The number of tasks
        """
        return len(self.tasks)