from __future__ import annotations

import asyncio
import enum
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.server import make_request
from hud.settings import settings
from hud.task import Task

if TYPE_CHECKING:
    from .adapters.common import Adapter

logger = logging.getLogger("hud.environment")


class Observation(BaseModel):
    """
    Observation from the environment.

    Attributes:
        screenshot: Base64 encoded PNG string of the screen
        text: Text observation, if available
    """

    screenshot: str | None = None  # base64 string png
    text: str | None = None


class TaskResult(BaseModel):
    """
    Result of a task step.

    Attributes:
        observation: The current observation
        reward: Reward value from the step
        terminated: Whether the task is complete
        info: Additional information from the environment
    """

    observation: Observation
    reward: float
    terminated: bool
    info: dict[str, Any]


class EnvironmentStatus(str, enum.Enum):
    """
    Status of the environment.

    Attributes:
        INITIALIZING: The environment is initializing
        RUNNING: The environment is running
        COMPLETED: The environment is completed
        ERROR: The environment is in an error state
    """

    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


status_messages = {
    EnvironmentStatus.RUNNING.value: "is running",
    EnvironmentStatus.ERROR.value: "had an error initializing",
    EnvironmentStatus.COMPLETED.value: "completed",
}


class Environment:
    """
    Environment interface for agent interactions.

    This class handles the environment state and interactions, including
    creating the environment, retrieving state, and executing actions.
    """

    def __init__(
        self,
        adapter: Adapter,
        run_id: str,
        id: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize an environment.

        Args:
            adapter: Adapter for converting actions
            run_id: ID of the run this environment belongs to
            id: Optional ID of an existing environment
            config: Optional configuration parameters
            metadata: Optional metadata for the environment
        """
        if metadata is None:
            metadata = {}
        if config is None:
            config = {}
        self.run_id = run_id
        self.config = config
        self.adapter = adapter
        self.metadata = metadata
        self.final_response: None | str = None
        self.id = id
        self.vnc_url = None

    async def create_environment(self) -> str:
        """
        Initialize the environment and return the task_run_id.

        Returns:
            str: The environment ID
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/create_environment",
            json={"run_id": self.run_id, "metadata": self.metadata},
            api_key=settings.api_key,
        )
        self.id = data["id"]
        return self.id

    async def get_vnc_url(self) -> str:
        """
        Get the VNC URL for the environment.

        Returns:
            str: The VNC URL for remote viewing/control
        """
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/environment/{self.id}/vnc",
            api_key=settings.api_key,
        )
        self.vnc_url = data["vm_url"]
        return self.vnc_url

    async def get_env_state(self) -> str:
        """
        Get the state of the environment.

        Returns:
            str: The current state (e.g., "running", "error")
        """
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/get_env_state/{self.id}",
            api_key=settings.api_key,
        )
        return data["state"]

    async def step(
        self, action: Any | None = None
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Send action to environment and get result.

        Args:
            action: The action to take, or None for no action

        Returns:
            tuple: (observation, reward, terminated, info)
        """
        action_list = self.translate_action(action) if action is not None else []
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/execute_step/{self.id}",
            json=action_list,
            api_key=settings.api_key,
        )
        # Convert the raw observation to the correct type
        self.current_observation = Observation(**data["observation"])
        data["observation"] = self.current_observation
        # Return the result
        task_result = TaskResult(**data)
        return (
            task_result.observation,
            task_result.reward,
            task_result.terminated,
            task_result.info,
        )

    def translate_action(self, action: Any) -> list:
        """
        Translate action to the correct format.

        Args:
            action: The action to translate

        Returns:
            list: List of translated actions in the CLA format
        """
        # Get adapter and then translate action to Common Language Action
        if isinstance(action, list):
            return self.adapter.adapt_list(action)
        return [self.adapter.adapt(action)]

    async def evaluate(self) -> float:
        """
        Get final evaluation score.

        Returns:
            float: The evaluation score
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/evaluation/{self.id}",
            api_key=settings.api_key,
        )
        return data["reward"]
    
    async def extract(self, extract: str | list[str | dict[str, Any]] | dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Extract some content from the environment.

        Args:
            extract: The extraction parameters

        Returns:
            dict: The extracted content
        """
        if extract is None:
            extract = {}
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.id}/extract",
            json=extract,
            api_key=settings.api_key,
        )
        return data

    async def execute(
        self, command_list: list[str]) -> dict[str, Any]:
        """
        Execute a command in the environment.
        Args:
            command_list: List of args
        Returns:
            dict["stdout"]: The standard output from the command
            dict["stderr"]: The standard error from the command
            dict["exit_code"]: The exit code from the command
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.id}/execute",
            json=command_list,
            api_key=settings.api_key,
        )
        return {
            "stdout": data["stdout"],
            "stderr": data["stderr"],
            "exit_code": data["exit_code"],
        }

    async def close(self) -> None:
        """
        Close the environment.
        """
        await make_request(
            method="POST",
            url=f"{settings.base_url}/close/{self.id}",
            api_key=settings.api_key,
        )

    async def reset(self, setup: str | list[str | dict[str, Any]] | dict[str, Any] | None = {}, task_id: str | None = None, metadata: dict[str, Any] | None = None) -> Observation:
        """
        Reset the environment to the task.

        Args:
            setup: Setup for the task
            task_id: ID of the task to reset to
            metadata: Optional metadata for the reset

        Returns:
            Observation: Initial observation for the task
        """
        if metadata is None:
            metadata = {}
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.id}/reset",
            json={"task_id": task_id, "setup": setup, "metadata": metadata},
            api_key=settings.api_key,
        )
        self.task_id = data["task_id"]
        return Observation(**data["observation"])

    async def wait_for_ready(self) -> None:
        """Wait for the environment to be ready"""
        while True:
            state = await self.get_env_state()
            if state in (
                EnvironmentStatus.RUNNING.value,
                EnvironmentStatus.ERROR.value,
                EnvironmentStatus.COMPLETED.value,
            ):
                logger.info("Environment %s %s", self.id, status_messages.get(state))
                break
            await asyncio.sleep(10)


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
        tasks: list[Task] | None = None,
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
        
    async def fetch_tasks(self) -> list[Task | str]:
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
