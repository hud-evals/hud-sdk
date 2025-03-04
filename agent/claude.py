import os
import json
from agent.base import Agent
from anthropic import Anthropic
from anthropic.types import Message

class ClaudeAgent(Agent):
    def __init__(self, client: Anthropic):
        super().__init__(client)
        self.model = "claude-3-7-sonnet-20250219"
        self.max_tokens = 4096
        self.tool_version = "20250124"
        self.thinking_budget = 1024
        self.conversation = []  # Store the full conversation history including Claude's responses

    async def predict(self, base64_image: str | None = None, input_text: str | None = None) -> tuple[bool, str | object | None]:
        message = self._create_message(base64_image, input_text)

        # Only append the message if it's not empty
        if message:
            self.conversation.append(message)

        response = self._generate_response()

        # Check if this response contains any tool_use blocks
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                break

        # Add Claude's response to the conversation history
        assistant_message = {"role": "assistant", "content": response.content}
        self.conversation.append(assistant_message)

        self.responses.append(response)

        done, processed = await self.process_response(response)

        return done, processed

    def _create_message(self, base64_image: str | None = None, input_text: str | None = None):
        """Create appropriate message based on context and inputs"""

        # Check if the previous response was from assistant and had tool_use
        if len(self.conversation) >= 2 and self.conversation[-1]["role"] == "assistant":
            last_assistant_message = self.conversation[-1]

            # Look for tool_use blocks in the assistant's message
            for block in last_assistant_message["content"]:
                if hasattr(block, "type") and block.type == "tool_use":
                    if hasattr(block, "name") and block.name == "computer" and base64_image:
                        # Found the tool_use to respond to
                        return {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": base64_image,
                                            },
                                        }
                                    ],
                                }
                            ],
                        }

        # Regular user message
        if input_text or base64_image:
            content = []
            if input_text:
                content.append({"type": "text", "text": input_text})
            if base64_image:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                    }
                )

            return {"role": "user", "content": content}

        return None  # Return None if no message could be created

    def _generate_response(self):
        beta_flag = (
            "computer-use-2025-01-24"
            if "20250124" in self.tool_version
            else "computer-use-2024-10-22"
        )

        tools = [
            {
                "type": f"computer_{self.tool_version}",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768,
                "display_number": 1,
            }
        ]

        thinking = {"type": "enabled", "budget_tokens": self.thinking_budget}

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=self.conversation,  # Use the full conversation including assistant responses
                tools=tools,
                betas=[beta_flag],
                thinking=thinking,
            )
            return response
        except Exception as e:
            raise

    async def process_response(self, response: Message) -> tuple[bool, str | object | None]:
        # Check if response contains a computer tool use
        computer_action = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "computer":
                computer_action = block.input
                break

        if response.content[-1].type == "text":
            # No computer tool use, treat as final response
            return True, str(response.content[-1].text)

        # If we have a computer action, adapt it to environment actions
        if computer_action:
            return False, computer_action

        return True, None
