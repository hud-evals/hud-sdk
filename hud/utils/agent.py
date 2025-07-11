from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hud.task import Task


def format_agent_prompt(environment_prompt: str | None, task: Task | None) -> str:
    """
    Format the agent prompt with the environment prompt and the task prompt.
    """
    prompt = ""

    # User-provided system prompt takes precedence over environment prompt
    if task and task.system_prompt:
        prompt += f"\n\n{task.system_prompt}"
    elif environment_prompt:
        prompt += f"\n\n{environment_prompt}"

    if task:
        if task.sensitive_data:
            prompt += "\n\nHere are placeholders for sensitive data for each domain:"
            for domain, credentials in task.sensitive_data.items():
                prompt += f"\n{domain}: "
                placeholders = [f"{key}" for key in credentials]
                prompt += f"{', '.join(placeholders)}"
            prompt += "\n\nYou can type these placeholders to enter the sensitive data when needed."

        if task.prompt:
            prompt += f"\n\n{task.prompt}"

    return prompt
