from __future__ import annotations

import json
from typing import Any, Union

# Define types for configuration
SetupConfig = Union[str, dict[str, Any], list[Any]]
EvaluateConfig = Union[str, dict[str, Any], list[Any]]

class Task:
    """A task that can be executed and evaluated.
    
    A Task represents a specific activity to be performed in an environment.
    It contains the gym ID to use, the prompt describing the task,
    and configurations for setting up and evaluating the environment.
    
    The setup and evaluate configurations can be in several formats:
    - String (function name): "chrome.maximize"
    - String (function with args): "chrome.activate_tab 5"
    - Dict: {"function": "chrome.navigate", "args": ["https://example.com"]}
    - List of the above: ["chrome.maximize", {"function": "chrome.navigate", "args": ["https://example.com"]}]
    
    Attributes:
        id: The remote task ID (optional)
        gym: The gym ID to load for this task
        prompt: The task prompt or instruction
        setup: Environment setup configuration
        evaluate: Configuration for evaluating responses
        metadata: Additional task metadata
    """
    
    def __init__(
        self,
        gym: str,
        prompt: str,
        setup: SetupConfig | None = None,
        evaluate: EvaluateConfig | None = None,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize a task.
        
        Args:
            gym: The gym ID to load
            prompt: The task prompt or instruction
            setup: Configuration for setting up the environment
            evaluate: Configuration for evaluating responses
            id: Optional remote ID of the task
            metadata: Additional task metadata
            **kwargs: Additional keyword arguments
        """
        self.id = id
        self.gym = gym
        self.prompt = prompt
        self.setup = setup
        self.evaluate = evaluate
        self.metadata = metadata or {}
        
        # Add any additional kwargs to metadata
        for key, value in kwargs.items():
            if key not in ["id", "gym", "prompt", "setup", "evaluate", "metadata"]:
                self.metadata[key] = value
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create a Task from a dictionary.
        
        Args:
            data: Dictionary containing task data
            
        Returns:
            Task instance
        """
        # Extract required fields
        gym = data.get("gym")
        if not gym:
            raise ValueError("Task requires a 'gym' field")
            
        prompt = data.get("prompt")
        if not prompt:
            raise ValueError("Task requires a 'prompt' field")
        
        # Create the task with all fields
        return cls(
            id=data.get("id"),
            gym=gym,
            prompt=prompt,
            setup=data.get("setup"),
            evaluate=data.get("evaluate"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> Task:
        """Create a Task from a JSON string.
        
        Args:
            json_str: JSON string containing task data
            
        Returns:
            Task instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
        
    def to_dict(self) -> dict[str, Any]:
        """Convert Task to a dictionary.
        
        Returns:
            Dictionary representation of the task
        """
        result = {
            "gym": self.gym,
            "prompt": self.prompt,
        }
        
        # Add optional fields if they exist
        if self.id:
            result["id"] = self.id
            
        if self.setup is not None:
            result["setup"] = json.dumps(self.setup)
            
        if self.evaluate is not None:
            result["evaluate"] = json.dumps(self.evaluate)
            
        if self.metadata:
            result["metadata"] = json.dumps(self.metadata)
            
        return result
    
    def to_json(self) -> str:
        """Convert Task to a JSON string.
        
        Returns:
            JSON string representation of the task
        """
        return json.dumps(self.to_dict())
    
    def __str__(self) -> str:
        return f"Task(gym={self.gym}, prompt={self.prompt})"
    
    def __repr__(self) -> str:
        return f"Task(id={self.id}, gym={self.gym}, prompt={self.prompt})"
