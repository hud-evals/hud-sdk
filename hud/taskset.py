from __future__ import annotations

from pathlib import PosixPath
from typing import TYPE_CHECKING, Any, get_args
from venv import logger

from pydantic import BaseModel

from hud.env.environment import create_remote_config
from hud.server import make_request
from hud.settings import settings
from hud.task import Task
from hud.types import CustomGym, ServerGym
from hud.utils.config import REMOTE_EVALUATE, REMOTE_SETUP

if TYPE_CHECKING:
    from collections.abc import Iterator

    from inspect_ai.dataset import Dataset

    from hud.agent import Agent


class TaskSet(BaseModel):
    """
    Collection of related tasks for benchmarking.

    Attributes:
        id: Unique identifier for the taskset
        name: Name of the taskset
        description: Description of the taskset
        tasks: List of Task objects in the taskset
    """

    id: str | None = None
    name: str | None = None
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

    async def upload(
        self,
        name: str | None = None,
        description: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Uploads the taskset to the server.
        """
        if name is None:
            name = self.name

        if name is None:
            raise ValueError("Taskset name is required")

        if api_key is None:
            api_key = settings.api_key

        # Convert all tasks to expanded configs
        processed_tasks = []
        for task in self.tasks:
            if task.setup is not None:
                setup_config = (
                    create_remote_config(None, task.setup, REMOTE_SETUP)[0].args[0].model_dump()
                )
            else:
                setup_config = None
            if task.evaluate is not None:
                evaluate_config = (
                    create_remote_config(None, task.evaluate, REMOTE_EVALUATE)[0]
                    .args[0]
                    .model_dump()
                )
            else:
                evaluate_config = None

            if isinstance(task.gym, CustomGym):
                if isinstance(task.gym.location, PosixPath):
                    raise ValueError(
                        "Local build contexts are not supported for "
                        "remote tasksets, attach an image or existing "
                        "gym id."
                    )
                gym_str = "docker"
                image_uri = task.gym.image_or_build_context
            elif isinstance(task.gym, str) and task.gym in get_args(ServerGym):
                gym_str = task.gym
                image_uri = None
            else:
                raise ValueError(f"Unknown gym type: {type(task.gym)}")

            processed_tasks.append(
                {
                    "prompt": task.prompt,
                    "gym": gym_str,
                    "setup": setup_config,
                    "evaluate": evaluate_config,
                    "config": task.config,
                    "image_uri": image_uri,
                    "description": task.description,
                }
            )

        await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/tasksets",
            api_key=api_key,
            json={
                "name": name,
                "description": description,
                "tasks": processed_tasks,
            },
        )
        logger.info(
            "Taskset %s uploaded successfully, see it on app.hud.so/evalsets/%s", name, name
        )

    def _apply(self, dict: dict[str, Any]) -> None:
        """
        Applies a parameter to all tasks in the taskset.
        """
        for task in self.tasks:
            for key, value in dict.items():
                setattr(task, key, value)

    def fit(self, agent: Agent | type[Agent]) -> None:
        """
        Automatically adapts the taskset to the agent's transfer_gyms.
        """
        if isinstance(agent, type):
            agent = agent()

        for task in self.tasks:
            if task.gym is None or isinstance(task.gym, CustomGym):
                continue
            task.gym = agent.transfer_gyms.get(task.gym, task.gym)


async def load_taskset(
    taskset_id: str,
    api_key: str | None = None,
    metadata: dict[str, Any] | None = None,
    load_custom_as_local: bool = False,
) -> TaskSet:
    """
    Loads a TaskSet by its ID.

    Args:
        taskset_id: The ID of the taskset to load
        api_key: Optional API key to use for the request
        metadata: Optional metadata to apply to the taskset
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

    logger.info(f"Taskset {taskset_id} loaded successfully")

    tasks = data["evalset"]
    for task in tasks:
        if task["gym"] == "docker":
            task["gym"] = CustomGym(
                location="local" if load_custom_as_local else "remote",
                image_or_build_context=task["image_uri"],
            )

    taskset = TaskSet.model_validate(
        {
            "id": taskset_id,
            "tasks": tasks,
        }
    )

    taskset._apply({"metadata": metadata})

    return taskset


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
