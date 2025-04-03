"""Base classes for environment implementations."""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import Any, Optional, Protocol, Union

from pydantic import BaseModel

from hud.task import EvaluateConfig, SetupConfig

logger = logging.getLogger("hud.environment")


class Observation(BaseModel):
    """
    Observation from the environment.

    Attributes:
        screenshot: Base64 encoded PNG string of the screen
        text: Text observation, if available
    """

    screenshot: Optional[str] = None  # base64 string png
    text: Optional[str] = None


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


def process_config(config: str | dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """
    Process a configuration into a standardized list of dictionary formats.
    
    Args:
        config: The configuration, which can be:
            - String (function name): "chrome.maximize"
            - String (function with args): "chrome.activate_tab 5"
            - Dict: {"function": "function_name", "args": [...]}
            - List of the above
            
    Returns:
        list[dict]: List of standardized configurations with function and args
        
    Raises:
        ValueError: If the configuration format is not recognized
    """
    logger.debug("Processing config: %s", config)
    
    # Handle list of configurations directly
    if isinstance(config, list):
        results = []
        for item in config:
            # Recursively process each item and extend results
            # (process_config always returns a list)
            results.extend(process_config(item))
        return results
    
    # Handle string configurations
    if isinstance(config, str):
        # Check if it's a simple function name or function with args
        parts = config.split(maxsplit=1)
        if len(parts) == 1:
            # Just a function name, no args
            return [{"function": parts[0], "args": []}]
        else:
            # Function with space-separated args
            func_name, args_str = parts
            return [{"function": func_name, "args": [args_str]}]
            
    # Handle dictionary configurations
    elif isinstance(config, dict) and "function" in config:
        func_name = config["function"]
        args = config.get("args", [])
        
        if not isinstance(args, list):
            # Single arg, convert to list
            args = [args]
            
        return [{"function": func_name, "args": args}]
        
    # Unknown configuration type
    error_msg = f"Unknown configuration type: {type(config)}"
    logger.error(error_msg)
    raise ValueError(error_msg)


class Environment(Protocol):
    """
    Environment protocol defining the interface for all environment implementations.
    
    This protocol defines the methods that all environment implementations must provide.
    Classes don't need to inherit from this protocol, but must implement all methods.
    """
    
    id: str
    metadata: dict[str, Any]
    url: str | None
    live_url: str | None
    
    async def create_environment(self) -> None:
        """Create and initialize the environment.
        
        This should be called after initialization to set up the environment.
        """
        ...
    
    def preload_setup(self, setup_config: SetupConfig) -> None:
        """Preload setup configuration from a Task.
        
        Args:
            setup_config: The setup configuration, which can be:
                - String (function name): "chrome.maximize"
                - String (function with args): "chrome.activate_tab 5"
                - Dict: {"function": [args]} where args are strings/ints/dicts
                - List of the above
        """
        ...
        
    def preload_evaluate(self, evaluate_config: EvaluateConfig) -> None:
        """Preload evaluation configuration from a Task.
        
        Args:
            evaluate_config: The evaluation configuration, which can be:
                - String (function name): "chrome.is_open"
                - String (function with args): "chrome.active_tab github.com"
                - Dict: {"function": [args]} where args are strings/ints/dicts
                - List of the above
        """
        ...
    
    async def setup(self, setup_config: Optional[SetupConfig] = None) -> Any:
        """Run a setup function in the environment.
        
        Args:
            setup_config: The setup configuration to run
            
        Returns:
            Any: Result of the setup function
        """
        ...
        
    async def step(self, command: str) -> Any:
        """Execute a step in the environment.
        
        Args:
            command: The command to execute
            
        Returns:
            Any: Result of the step execution
        """
        ...
        
    async def evaluate(self, evaluate_config: Optional[EvaluateConfig] = None) -> Any:
        """Run an evaluation function in the environment.
        
        Args:
            evaluate_config: The evaluation configuration to run
            
        Returns:
            Any: Result of the evaluation function
        """
        ...
        
    async def get_info(self, function_name: str = "get_state", *args: Any, **kwargs: Any) -> Any:
        """Get information from the environment.
        
        Args:
            function_name: The name of the info function to run (default: "get_state")
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Result of the info function
        """
        ...
        
    async def execute(self, command: str) -> dict[str, Any]:
        """Execute a command in the environment.
        
        Args:
            command: The command to execute
            
        Returns:
            dict: Results with stdout, stderr, and exit_code
        """
        ...
    
    async def get_urls(self) -> dict[str, str]:
        """Get URLs for the environment.
        
        Returns:
            dict: Dictionary of URLs for accessing the environment
        """
        ...
    
    async def wait_for_ready(self) -> None:
        """Wait for the environment to be ready.
        
        This method should check if the environment is in a ready state
        and wait until it is.
        """
        ...
            
    async def close(self) -> None:
        """Close the environment.
        
        This should release any resources and clean up the environment.
        """
        ...

