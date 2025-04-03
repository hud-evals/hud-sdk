from __future__ import annotations

from typing import Optional
from hud.environment import settings
from hud.server import make_request
from hud.task import Task


async def fetch_task_ids(evalset_id: str, *, api_key: Optional[str] = None) -> list[str]:
    """
    Fetch all task IDs in this evalset from the API.
    
    Returns:
        list[str]: List of task IDs
    """
    
    if api_key is None:
        api_key = settings.api_key
    
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/evalsets/{evalset_id}/tasks",
        api_key=api_key,
    )
    return [task_id for task_id in data["tasks"]]
        
async def load_evalset(evalset_id: str, api_key: Optional[str] = None ) -> EvalSet:
    """
    Fetch all tasks in this evalset from the API.

    Returns:
        EvalSet: List of tasks or their IDs in the database
    """
    
    if api_key is None:
        api_key = settings.api_key
    
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/evalsets/{evalset_id}/tasks",
        api_key=api_key,
    )
    tasks = []
    if "evalset" in data:
        for task in data["evalset"]:
            tasks.append(Task(**task))
    else:
        for task in data["tasks"]:
            tasks.append(Task(id=task))
    
    return EvalSet(
        id=evalset_id,
        tasks=tasks,
    )

class EvalSet:
    """
    Evaluation set containing tasks for benchmarking.

    Attributes:
        id: Unique identifier for the evalset
        tasks: List of task IDs in this evalset
    """

    def __init__(
        self,
        id: str,
        tasks: list[Task],
    ) -> None:
        """
        Initialize an evaluation set.

        Args:
            id: Unique identifier
            name: Human-readable name
            tasks: Optional list of task IDs
        """
        self.id = id
        self.tasks = tasks

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