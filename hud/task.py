from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from inspect_ai.util._sandbox import SandboxEnvironmentSpec
from pydantic import BaseModel, Field

from hud.types import CustomGym, Gym, MetadataKeys, SensitiveData
from hud.utils.common import FunctionConfig, FunctionConfigs

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample

    from hud.agent import Agent


def convert_inspect_setup(setup: str) -> list[FunctionConfig]:
    """
    Inspect setup is a single bash string to run in the environment.
    We convert this into a single FunctionConfig using the exec command
    """
    return [FunctionConfig(function="bash", args=[setup])]


class Task(BaseModel):
    """A task that can be executed and evaluated.

    A Task represents a specific activity to be performed in an environment.
    It contains the prompt describing the task and configurations for
    setting up and evaluating the environment.

    The setup and evaluate configurations can be in several formats:
    - String (function name): "chrome.maximize"
    - Tuple (function with args): ("chrome.activate_tab", 5)
    - Dict: {"function": "chrome.navigate", "args": ["https://example.com"]}
    - List of the above: ["chrome.maximize", {"function": "chrome.navigate", "args": ["https://example.com"]}]

    Attributes:
        id: The remote task ID (optional if local-only)
        prompt: The task prompt or instruction
        system_prompt: The system prompt for the evalset (optional)
        setup: Environment setup configuration (optional)
        evaluate: Configuration for evaluating responses
        metadata: Additional task metadata
        sensitive_data: Sensitive data such as API keys, passwords, etc.
        choices: Multiple choice answer list (for Inspect compatibility)
        target: Ideal target output (for Inspect compatibility)
        files: Files that go along with the task (for Inspect compatibility)
        gym: Environment specification
    """

    id: str | None = None  # Remote task ID (optional if local-only)

    prompt: str  # Task prompt or instruction
    system_prompt: str | None = None  # System prompt for the evalset (optional)

    gym: Gym | None = None  # Environment specification

    # Setup and evaluate configurations for the environment (environment specific)
    setup: FunctionConfigs | None = None
    evaluate: FunctionConfigs | None = None

    # Overflow configuration for environments that don't conform to the standard
    config: dict[str, Any] | None = None

    # Sensitive data such as API keys, passwords, etc.
    sensitive_data: SensitiveData = Field(default_factory=dict)

    # Metadata for the task evaluation, information about the agent (see MetadataKeys)
    metadata: dict[MetadataKeys, Any] = Field(default_factory=dict)

    # Description of the task, for extra information about its purpose and context
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        # Handle CustomGym conversion if needed
        gym_data = data.get("gym")
        if (
            isinstance(gym_data, dict)
            and gym_data.get("type") == "public"
            and gym_data.get("location") in ("local", "remote")
            and gym_data.get("image_or_build_context") is not None
        ):
            data = data.copy()  # Don't modify the original
            # Preserve strings as image names, only convert paths to Path objects
            image_or_build_context = gym_data["image_or_build_context"]
            if isinstance(image_or_build_context, str) and not image_or_build_context.startswith('/'):
                # Keep as string for Docker image names
                parsed_image_or_build_context = image_or_build_context
            else:
                # Convert to Path for filesystem paths
                parsed_image_or_build_context = Path(image_or_build_context)
            
            data["gym"] = CustomGym(
                type=cast("Literal['public']", gym_data["type"]),
                location=cast("Literal['local', 'remote']", gym_data["location"]),
                image_or_build_context=parsed_image_or_build_context,
                host_config=gym_data.get("host_config"),
            )
        
        return cls(**data)

    @classmethod
    def from_serialized(cls, data: dict[str, Any]) -> Task:
        gym_data = data.get("gym")
        parsed_gym: Gym | None = gym_data

        parsed_setup = [(param, entry) for param, entry in data.get("setup", [])]
        parsed_evaluate = [(param, entry) for param, entry in data.get("evaluate", [])]

        # Convert dict gym data to CustomGym if needed
        if (
            isinstance(gym_data, dict)
            and gym_data.get("type") == "public"
            and gym_data.get("location") in ("local", "remote")
            and gym_data.get("image_or_build_context") is not None
        ):
            # Preserve strings as image names, only convert paths to Path objects
            image_or_build_context = gym_data["image_or_build_context"]
            if isinstance(image_or_build_context, str) and not image_or_build_context.startswith('/'):
                # Keep as string for Docker image names
                parsed_image_or_build_context = image_or_build_context
            else:
                # Convert to Path for filesystem paths
                parsed_image_or_build_context = Path(image_or_build_context)
            
            parsed_gym = CustomGym(
                type=cast("Literal['public']", gym_data["type"]),
                location=cast("Literal['local', 'remote']", gym_data["location"]),
                image_or_build_context=parsed_image_or_build_context,
            )

        return cls(
            id=data.get("id"),
            prompt=data.get("prompt", ""),
            system_prompt=data.get("system_prompt"),
            setup=parsed_setup,
            evaluate=parsed_evaluate,
            gym=parsed_gym,
            config=data.get("config"),
            description=data.get("description"),
            sensitive_data=data.get("sensitive_data", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_inspect_sample(cls, sample: Sample) -> Task:
        """Create a Task from an Inspect dataset sample.
        Automatically detects if a CustomGym (docker) or QA Gym is needed based on sample.sandbox.
        Configures evaluation using 'response_includes' or 'match_all' based on sample.target.

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
        prompt = sample.input
        if isinstance(prompt, list):
            prompt_parts = []
            for message in prompt:
                role = message.role
                content = message.content
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt = "\n\n".join(prompt_parts)

        evaluate_config = None
        if sample.target:
            if isinstance(sample.target, str):
                evaluate_config = FunctionConfig(function="response_includes", args=[sample.target])
            elif isinstance(sample.target, list):
                evaluate_config = FunctionConfig(function="match_all", args=sample.target)

        task_setup: FunctionConfigs | None = (
            convert_inspect_setup(sample.setup) if sample.setup else None
        )

        sandbox = sample.sandbox

        match sandbox:
            case "docker":
                task_gym = CustomGym(
                    image_or_build_context="ubuntu:latest",
                    location="local",
                )
            case SandboxEnvironmentSpec(type="docker", config=str()):
                # create temp dir and put dockerfile there, then use that path
                temp_dir = tempfile.mkdtemp()
                temp_dir_path = Path(temp_dir)
                dockerfile_path = temp_dir_path / "Dockerfile"
                dockerfile_path.write_text(sandbox.config)
                task_gym = CustomGym(
                    image_or_build_context=temp_dir_path,
                    location="local",
                )
            case None:
                task_gym = "qa"
                task_setup = None
            case _:
                raise ValueError(f"Unsupported sandbox type: {sandbox}")

        return cls(
            id=None,
            prompt=prompt,
            setup=task_setup,
            evaluate=evaluate_config,
            gym=task_gym,
            # files=sample.files, # TODO: Decide how/if to handle files
        )

    async def fit(self, agent: Agent | type[Agent]) -> None:
        if isinstance(agent, type):
            agent = agent()

        if self.gym is None:
            return
        self.gym = agent.transfer_gyms.get(self.gym, self.gym)

    def serialize(self) -> dict[str, Any]:
        if isinstance(self.setup, list):
            parsed_setup = [[param, entry] for param, entry in self.setup]
        else:
            parsed_setup = self.setup
        if isinstance(self.evaluate, list):
            parsed_evaluate = [[param, entry] for param, entry in self.evaluate]
        else:
            parsed_evaluate = self.evaluate

        if isinstance(self.gym, CustomGym):
            parsed_gym = self.gym.model_dump()
            parsed_gym["image_or_build_context"] = str(parsed_gym["image_or_build_context"])
        else:  # is ServerGym
            parsed_gym = self.gym

        return {
            "id": self.id,
            "prompt": self.prompt,
            "config": self.config,
            "description": self.description,
            "setup": parsed_setup,
            "evaluate": parsed_evaluate,
            "gym": parsed_gym,
            "sensitive_data": self.sensitive_data,
            "metadata": self.metadata,
        }
