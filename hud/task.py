from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.types import CustomGym, Gym
from hud.utils.common import HudStyleConfig, HudStyleConfigs

if TYPE_CHECKING:
    from inspect_ai.dataset import Sample

# Environment specifications:
# These represent the environment as a whole, including both the controller
# and the environment type (eg, what os, which services are running)

UBUNTU_DOCKERFILE = "ubuntu:latest"


def convert_inspect_setup(setup: str) -> list[HudStyleConfig]:
    """
    Inspect setup is a single bash string to run in the environment.
    We convert this into a single HudStyleConfig using the exec command
    """
    return [HudStyleConfig(function="bash", args=[setup])]


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
    setup: HudStyleConfigs | None = None
    evaluate: HudStyleConfigs | None = None
    gym: Gym | None = None
    
    target: str | list[str] | None = None
    
    choices: list[str] | None = None
    files: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    
    config: dict[str, Any] | None = None

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
                evaluate_config = ("response_includes", [sample.target])
            elif isinstance(sample.target, list):
                evaluate_config = ("match_all", sample.target)

        task_gym: Gym | None = None
        task_setup: HudStyleConfigs | None = None
        
        sandbox = sample.sandbox
        dockerfile = None
        use_qa_gym = True

        if sandbox:
            if isinstance(sandbox, str):
                if sandbox == "docker":
                    dockerfile = UBUNTU_DOCKERFILE
                    use_qa_gym = False
            elif isinstance(sandbox, tuple) and len(sandbox) == 2:
                sandbox_type, sandbox_config = sandbox
                if sandbox_type == "docker":
                    dockerfile = sandbox_config
                    use_qa_gym = False

        if use_qa_gym:
            task_gym = "qa"
            task_setup = None
        else:
            task_gym = CustomGym(
                dockerfile=dockerfile or UBUNTU_DOCKERFILE,
                location="local",
            )
            task_setup = [x for x in convert_inspect_setup(sample.setup)] if sample.setup else None
            # TODO: Handle sample.files for CustomGym case if needed


        return cls(
            id=None,
            prompt=prompt,
            setup=task_setup,
            metadata=sample.metadata,
            choices=sample.choices,
            evaluate=evaluate_config,
            gym=task_gym,
            # files=sample.files, # TODO: Decide how/if to handle files
        )
