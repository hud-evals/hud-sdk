import json
import logging
import os
from typing import Any, Literal, cast

from openai import AsyncOpenAI
from openai.types.responses import (
    ToolParam,
    ResponseInputParam,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseComputerToolCall,
    ResponseOutputText,
)

from hud.adapters import Adapter
from hud.agent.base import Agent
from hud.adapters.operator import OperatorAdapter
from hud.types import Gym
from hud.utils.common import Observation
from hud.settings import settings
from hud.adapters.common.types import LogType

logger = logging.getLogger(__name__)


class OperatorAgent(Agent[AsyncOpenAI, dict[str, Any]]):
    """
    An agent implementation using OpenAI's Computer Use API.

    This agent interacts with HUD environments using OpenAI's Computer Use API
    through the OperatorAdapter which converts actions to the format expected by HUD.
    """

    transfer_gyms: dict[Gym, Gym] = {"qa": "hud-browser"}

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str = "computer-use-preview",
        environment: Literal["windows", "mac", "linux", "browser"] = "linux",
        adapter: Adapter | None = None,
        max_iterations: int = 8,
        name: str | None = None,
    ):
        """
        Initialize the OperatorAgent.

        Args:
            client: The AsyncOpenAI client for API calls (optional, created automatically if not provided)
            model: The model to use for computer use
            environment: The environment type (windows, mac, linux, browser)
            adapter: The adapter to use for preprocessing and postprocessing
            max_iterations: Maximum number of iterations for the agent
            name: The name of the agent
        """
        # Initialize client if not provided
        if client is None:
            # Get API key from settings
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found in settings or environment variables. Set OPENAI_API_KEY."
                )

            # Create asynchronous client
            client = AsyncOpenAI(api_key=api_key)

        adapter = adapter or OperatorAdapter()

        if name is None:
            name = f"openai-{model}"

        super().__init__(client=client, adapter=adapter, name=name)

        self.model = model
        self.environment = environment
        self.max_iterations = max_iterations

        # Default dimensions
        self.width = 1024
        self.height = 768

        # Update dimensions if adapter is provided
        if self.adapter:
            self.width = self.adapter.agent_width
            self.height = self.adapter.agent_height

        # Message history and state tracking
        self.last_response_id = None
        self.pending_call_id = None
        self.initial_prompt = None
        self.pending_safety_checks = []

    async def fetch_response(self, observation: Observation) -> tuple[list[dict[str, Any]], bool]:
        """
        Fetch a response from the model based on the observation.

        Args:
            observation: The preprocessed observation

        Returns:
            tuple[list[dict[str, Any]], bool, list[LogType] | None]: A tuple containing the list of raw actions,
                                             boolean indicating if the agent believes the task is complete.
        """
        if not self.client:
            raise ValueError("Client is required")

        # Define the computer use tool with correct type using cast
        computer_tool = cast(
            ToolParam,
            {
                "type": "computer_use_preview",
                "display_width": self.width,
                "display_height": self.height,
                "environment": self.environment,
            },
        )

        # Process the observation based on whether it's the first one or a response to an action
        if self.pending_call_id is None and self.last_response_id is None:
            # This is the first observation, store and send the prompt
            self.initial_prompt = observation.text

            # Create the initial request following the required structure
            input_content: list[dict[str, Any]] = [
                {"type": "input_text", "text": observation.text or ""}
            ]

            # Add screenshot if present
            if observation.screenshot:
                input_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{observation.screenshot}",
                    }
                )

            # Structure the input correctly for the API using cast
            input_param = cast(ResponseInputParam, [{"role": "user", "content": input_content}])

            # Call OpenAI API for the initial prompt (asynchronous call)
            response = await self.client.responses.create(
                model=self.model,
                tools=[computer_tool],
                input=input_param,
                truncation="auto",
                reasoning={"summary": "auto"},
            )

        else:
            # This is a response to a previous action
            if not observation.screenshot:
                logger.warning("No screenshot provided for response to action")
                return [], True

            # Create a response to the previous action with the new screenshot
            input_param_followup = cast(
                ResponseInputParam,
                [
                    cast(
                        ResponseInputItemParam,
                        {
                            "call_id": self.pending_call_id,
                            "type": "computer_call_output",
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{observation.screenshot}",
                            },
                            "acknowledged_safety_checks": self.pending_safety_checks,
                        },
                    )
                ],
            )
            self.pending_safety_checks = []

            # Call OpenAI API for follow-up (asynchronous call)
            response = await self.client.responses.create(
                model=self.model,
                previous_response_id=self.last_response_id,
                tools=[computer_tool],
                input=input_param_followup,
                truncation="auto",
            )

        # Store the response ID for the next call
        self.last_response_id = response.id

        # Process the response to extract actions or final text
        actions = []
        done = True  # Assume done unless a computer call is found
        final_text_response = ""

        # Check for computer calls first
        computer_calls = [
            item
            for item in response.output
            if isinstance(item, ResponseComputerToolCall) and item.type == "computer_call"
        ]

        if computer_calls:
            # If computer calls exist, process them and set done=False
            done = False
            for computer_call in computer_calls:
                self.pending_call_id = computer_call.call_id
                action = computer_call.action
                self.pending_safety_checks = computer_call.pending_safety_checks
                actions.append(action.model_dump())  # Convert Pydantic model to dict
                # logger.info(f"Computer call action: {action}")
        else:
            # No computer calls, check for a final text message
            # logger.info("No computer call found. Checking for final message.")
            # logger.info(response.output)
            for item in response.output:
                if isinstance(item, ResponseOutputMessage) and item.type == "message":
                    # Extract text from content blocks within the message
                    full_text = "".join(
                        [c.text for c in item.content if isinstance(c, ResponseOutputText)]
                    )
                    if full_text:
                        final_text_response = full_text
                        # logger.info(f"Final text message: {final_text_response}")
                        break  # Stop after finding the first text message

            # If we found final text, package it as a 'response' action
            if final_text_response:
                # No ResponseAgent logic here anymore - just return the response
                actions = [{"type": "response", "text": final_text_response}]
                done = True
            else:
                logger.info("No computer calls and no final text message found.")
            # Keep done = True, actions remains empty

        reasoning = ""
        for item in response.output:
            if item.type == "reasoning":
                reasoning += f"Thinking: {item.summary[0].text}\n"
            elif item.type == "message":
                for content in item.content:
                    if isinstance(content, ResponseOutputText):
                        reasoning += f"{content.text}\n"

        # add reasoning to the actions
        for action in actions:
            action["reasoning"] = reasoning
            action["logs"] = response.model_dump()  # type: ignore[assignment]

        return actions, done
