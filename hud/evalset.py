

from typing import Optional
from hud.environment import settings
from hud.server import make_request
from hud.task import Task


class EvalSet:
    """
    Evaluation set containing tasks for benchmarking.

    Attributes:
        id: Unique identifier for the evalset
        name: Human-readable name
        tasks: List of task IDs in this evalset
    """

    def __init__(
        self,
        id: str,
        name: str,
        tasks: Optional[list[Task]] = None,
    ) -> None:
        """
        Initialize an evaluation set.

        Args:
            id: Unique identifier
            name: Human-readable name
            tasks: Optional list of task IDs
        """
        self.id = id
        self.name = name
        self.tasks = tasks or []

    def __getitem__(self, index: int) -> Task:
        """
        Get task by index.
        
        Args:
            index: Index of the task
            
        Returns:
            Task ID at the specified index
        """
        return self.tasks[index]
        
    def __len__(self) -> int:
        """
        Get the number of tasks.
        
        Returns:
            Number of tasks in the evaluation set
        """
        return len(self.tasks)
    
    async def fetch_task_ids(self) -> list[str]:
        """
        Fetch all task IDs in this evalset from the API.
        
        Returns:
            list[str]: List of task IDs
        """
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/evalsets/{self.id}/tasks",
            api_key=settings.api_key,
        )
        return [task_id for task_id in data["tasks"]]
        
    async def fetch_tasks(self) -> list[Task]:
        """
        Fetch all tasks in this evalset from the API.

        Returns:
            list[Task | str]: List of tasks or their IDs in the database
        """
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/evalsets/{self.id}/tasks",
            api_key=settings.api_key,
        )
        self.tasks = []
        if "evalset" in data:
            for task in data["evalset"]:
                self.tasks.append(Task(**task))
        else:
            for task in data["tasks"]:
                self.tasks.append(Task(id=task))
        return self.tasks
