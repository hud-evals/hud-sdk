from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from inspect_ai.util._sandbox import SandboxEnvironmentSpec
from pydantic import BaseModel

from hud.types import CustomGym, Gym, MetadataKeys
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
        setup: Environment setup configuration (optional)
        evaluate: Configuration for evaluating responses
        metadata: Additional task metadata
        choices: Multiple choice answer list (for Inspect compatibility)
        target: Ideal target output (for Inspect compatibility)
        files: Files that go along with the task (for Inspect compatibility)
        gym: Environment specification
    """

    id: str | None = None
    prompt: str
    setup: FunctionConfigs | None = None
    evaluate: FunctionConfigs | None = None
    gym: Gym | None = None
    config: dict[str, Any] | None = None

    metadata: dict[MetadataKeys, Any] | None = None

    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
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
            parsed_gym = CustomGym(
                type=cast("Literal['public']", gym_data["type"]),
                location=cast("Literal['local', 'remote']", gym_data["location"]),
                image_or_build_context=Path(gym_data["image_or_build_context"]),
            )

        return cls(
            id=data.get("id"),
            prompt=data.get("prompt", ""),
            setup=parsed_setup,
            evaluate=parsed_evaluate,
            gym=parsed_gym,
            config=data.get("config"),
            description=data.get("description"),
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
        }
