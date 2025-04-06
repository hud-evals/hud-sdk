"""RemoteEnvironment implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from hud.adapters.common.types import CLA
from hud.server import make_request
from hud.settings import settings

from .base import (
    Environment,
    EnvironmentStatus,
    EvaluateConfig,
    Observation,
    SetupConfig,
    TaskResult,
    process_config,
    status_messages,
)

logger = logging.getLogger("hud.environment")


class RemoteEnvironment(Environment):
    """
    Environment interface for remote agent interactions.
    
    This class handles interactions with a remote environment, including
    creating the environment, retrieving state, and executing actions.
    """
    
    def __init__(
        self,
        gym_id: str | None = None,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize an environment.
        
        Args:
            id: Optional ID of an existing environment
            metadata: Optional metadata for the environment
        """
        self.id = id or ""
        self.metadata = metadata or {}
        self.url = None
        self.live_url = None
        self.gym_id = gym_id

        # For preloaded setup and evaluate configurations
        self._preloaded_setup = None
        self._preloaded_evaluate = None
    
    def preload_setup(self, setup_config: SetupConfig) -> None:
        """Preload setup configuration from a Task.
        
        Args:
            setup_config: The setup configuration
        """
        logger.debug("Preloading setup configuration: %s", setup_config)
        self._preloaded_setup = setup_config
        
    def preload_evaluate(self, evaluate_config: EvaluateConfig) -> None:
        """Preload evaluation configuration from a Task.
        
        Args:
            evaluate_config: The evaluation configuration
        """
        logger.debug("Preloading evaluate configuration: %s", evaluate_config)
        self._preloaded_evaluate = evaluate_config
    
    async def create_environment(self) -> str:
        """Create the environment.
        
        Creates a new environment on the remote server.
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/create_environment",
            json={"metadata": self.metadata, "gym_id": self.gym_id},
            api_key=settings.api_key,
        )
        self.id = data["id"]
        return self.id
    
    async def get_vnc_url(self) -> str | None:
        """
        Get the VNC URL for the environment.
        
        Returns:
            str: The VNC URL for remote viewing/control
        """
        return self.live_url
    
    async def get_urls(self) -> dict[str, Any]:
        """Get URLs for the environment.
        
        Returns:
            dict: Dictionary of URLs for accessing the environment
        """
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/environment/{self.id}/urls",
            api_key=settings.api_key,
        )
        
        self.url = data.get("url")
        self.live_url = data.get("live_url")
        
        return {
            "url": self.url,
            "live_url": self.live_url,
        }
    
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
        if "urls" in data:
            self.url = data["urls"]["url"]
            self.live_url = data["urls"]["live_url"]
        return data["state"]
    
    async def setup(self, setup_config: SetupConfig | None = None) -> Any:
        """Run a setup function in the environment.
        
        Args:
            setup_config: The setup configuration to run
            
        Returns:
            Any: Result of the setup function
        """
        if not self.id:
            logger.warning("No environment ID to run setup for")
            raise ValueError("No environment ID")
            
        # If no config provided and we have preloaded config, use that
        if setup_config is None and self._preloaded_setup is not None:
            setup_config = self._preloaded_setup
        elif setup_config is None:
            raise ValueError("No setup configuration provided and no preloaded setup configuration")
            
        logger.debug("Processing setup configuration: %s", setup_config)
        processed_configs = process_config(setup_config)
            
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.id}/reset",
            json={"setup": processed_configs},
            api_key=settings.api_key,
        )
        self.task_id = data["task_id"]
        return Observation(**data["observation"])
    
    async def step(
        self, action: CLA | list[CLA]
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Send action to environment and get result.
        
        Args:
            action: The action to take, or None for no action
            
        Returns:
            tuple: (observation, reward, terminated, info)
        """
        
        action_list = action if isinstance(action, list) else [action]
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
    
    async def evaluate(self, evaluate_config: EvaluateConfig | None = None) -> Any:
        """Run an evaluation function in the environment.
        
        Args:
            evaluate_config: The evaluation configuration to run
            
        Returns:
            Any: Result of the evaluation function
        """
        if not self.id:
            logger.warning("No environment ID to run evaluation for")
            raise ValueError("No environment ID")
            
        # If no config provided and we have preloaded config, use that
        if evaluate_config is None and self._preloaded_evaluate is not None:
            evaluate_config = self._preloaded_evaluate
        elif evaluate_config is None:
            raise ValueError("No evaluation configuration provided and no preloaded evaluation configuration")
            
        logger.debug("Processing evaluation configuration: %s", evaluate_config)
        processed_configs = process_config(evaluate_config)
        
        return await make_request(
            method="POST",
            url=f"{settings.base_url}/evaluation/{self.id}",
            json={"evaluate": processed_configs},
            api_key=settings.api_key,
        )
    
    async def get_info(self, function_name: str = "get_state", *args: Any, **kwargs: Any) -> Any:
        """
        Get information from the environment.
        
        Args:
            function_name: The name of the info function to run (default: "get_state")
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Result of the info function
        """
        # Convert kwargs to args for API compatibility
        all_args = list(args)
        if kwargs:
            all_args.append(kwargs)
            
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/environment/{self.id}/info",
            json={"function": function_name, "args": all_args},
            api_key=settings.api_key,
        )
        return data
    
    async def execute(
        self, command_list: list[str]
    ) -> dict[str, Any]:
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
            "stdout": b64decode(data["stdout"]),
            "stderr": b64decode(data["stderr"]),
            "exit_code": data["exit_code"],
        }
    
    async def copy_from(
        self, path: str
    ) -> bytes:
        """
        Copy a file from the environment to the client.
        
        Args:
            path: Path to the file to copy
            
        Returns:
            bytes: Content of the file
        """
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.id}/copy_from",
            json={"path": path},
            api_key=settings.api_key,
        )
        
        # convert from base64 to bytes
        return b64decode(data["content"])
        
    async def copy_to(
        self, 
        path: str,
        content: bytes,
    ) -> None:
        """
        Copy a file to the environment.
        
        Args:
            path: Path to the file
            content: Content of the file
            
        Returns:
            None
        """
        await make_request(
            method="POST",
            url=f"{settings.base_url}/environments/{self.id}/copy_to",
            json= {
                "path": path,
                "content": b64encode(content).decode("utf-8"),
            },
            api_key=settings.api_key,
        )
    
    async def wait_for_ready(self) -> None:
        """Wait for the environment to be ready."""
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
            
    async def close(self) -> None:
        """
        Close the environment.
        """
        await make_request(
            method="POST",
            url=f"{settings.base_url}/close/{self.id}",
            api_key=settings.api_key,
        )
