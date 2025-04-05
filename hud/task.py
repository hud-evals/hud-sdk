from __future__ import annotations

import json
from typing import Any, Optional, Union, List, Dict
from pathlib import Path
from typing import Literal, Union
import uuid
from inspect_ai.dataset import Sample
from pydantic import BaseModel

# Environment specifications:
# These represent the environment as a whole, including both the controller and the environment type (eg, what os, which services are running)

UBUNTU_DOCKERFILE = "ubuntu:latest"

class PrivateEnvSpec(BaseModel):
    """Private environment specification identified by an id."""
    type: Literal["private"] = "private"
    id: str


class PublicEnvSpec(BaseModel):
    """
    Public environment specification with a dockerfile and controller.
    
    If the controller is remote, the env will be created on the server.
    If the controller is dev, the env will be created locally via docker.
    """
    type: Literal["public"] = "public"
    dockerfile: str
    location: Literal["local", "remote"]
    # If path, then it is a development environment on the local computer
    # If str, then it is an environment id (which is a URL to a .tar file)
    controller: Union[Path, str]

# permits either a private or public environment specification
EnvSpec = Union[PrivateEnvSpec, PublicEnvSpec]

class Task(BaseModel):
    """A task that can be executed and evaluated.
    
    A Task represents a specific activity to be performed in an environment.
    It contains the prompt describing the task and configurations for
    setting up and evaluating the environment.
    
    The setup and evaluate configurations can be in several formats:
    - String (function name): "chrome.maximize"
    - String (function with args): "chrome.activate_tab 5"
    - Dict: {"function": "chrome.navigate", "args": ["https://example.com"]}
    - List of the above: ["chrome.maximize", {"function": "chrome.navigate", "args": ["https://example.com"]}]
    
    Attributes:
        id: The remote task ID (optional if local-only)
        prompt: The task prompt or instruction
        setup: Environment setup configuration (optional)
        evaluate: Configuration for evaluating responses
        metadata: Additional task metadata
        choices: Multiple choice answer list (for Inspect compatibility)
        target: Ideal target output (for Inspect compatibility)
        files: Files that go along with the task (for Inspect compatibility)
        envspec: Environment specification (for Inspect compatibility)
    """
    id: Optional[str] = None
    prompt: str
    setup: Optional[list[str]] = None
    evaluate: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    choices: Optional[List[str]] = None
    target: Optional[Union[str, List[str]]] = None
    files: Optional[Dict[str, str]] = None
    envspec: Optional[EnvSpec] = None
    
    @classmethod
    def from_inspect_sample(cls, sample: Sample) -> Task:
        """Create a Task from an Inspect dataset sample.
        The task's sandbox is a local ubuntu container using the standard controller.
        Files will be copied to the user directory
        
        Args:
            sample: An Inspect dataset Sample object
            
        Returns:
            Task instance
        
        The Inspect Sample has these fields:
        - input (str | list[ChatMessage]): The input to be submitted to the model
        - choices (list[str] | None): Optional multiple choice answer list
        - target (str | list[str] | None): Optional ideal target output
        - id (str | None): Optional unique identifier for sample
        - metadata (dict[str, Any] | None): Optional arbitrary metadata
        - sandbox (str | tuple[str, str]): Optional sandbox environment type
        - files (dict[str, str] | None): Optional files that go with the sample
        - setup (str | None): Optional setup script to run for sample
        """
        # Extract the input as prompt
        prompt = sample.input
        if isinstance(prompt, list):  # Handle ChatMessage format
            # Convert chat message list to a string representation
            prompt_parts = []
            for message in prompt:
                role = message.role
                content = message.content
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt = "\n\n".join(prompt_parts)
        
        
        # Map sandbox from Inspect to our envspec
        sandbox = sample.sandbox
        dockerfile = None
        if sandbox:
            if isinstance(sandbox, str):
                assert sandbox == "docker", "docker is the only supported sandbox"
            elif isinstance(sandbox, tuple) and len(sandbox) == 2:
                sandbox_type, sandbox_config = sandbox
                assert sandbox_type == "docker", "docker is the only supported sandbox"
                dockerfile = sandbox_config
            else:
                raise ValueError(f"Invalid sandbox format: {sandbox}")


        envspec = PublicEnvSpec(
            dockerfile=dockerfile or UBUNTU_DOCKERFILE,
            location="local",
            controller="https://hud.so/registry/controllers/hud/ubuntu.tar"
        )        
        
        return cls(
            id=str(sample.id) if sample.id else None,
            prompt=prompt,
            setup=[sample.setup] if sample.setup else None,
            metadata=sample.metadata,
            choices=sample.choices,
            target=sample.target,
            envspec=envspec
        )
    
    