import logging
from typing import Any, List, Optional, Tuple, cast, Dict, Union, Literal, TypedDict

from anthropic import Anthropic
from anthropic.types import Message, ToolUseBlock, ContentBlock
from anthropic.types.beta import (
    BetaMessageParam,
    BetaToolParam,
    BetaToolResultBlockParam,
    BetaToolComputerUse20250124Param,
    BetaBase64ImageSourceParam,
    BetaTextBlockParam,
    BetaImageBlockParam,
)


from agent.base import Agent
from hud.adapters.claude.adapter import ClaudeAdapter
from hud.adapters.common.types import CLA
from hud.env.environment import Observation

logger = logging.getLogger(__name__)

def base64_to_content_block(base64: str) -> BetaImageBlockParam:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64
        }
    }

def text_to_content_block(text: str) -> BetaTextBlockParam:
    return {
        "type": "text",
        "text": text
    }

def tool_use_content_block(tool_use_id: str, content: list[BetaTextBlockParam | BetaImageBlockParam]) -> BetaToolResultBlockParam:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content
    }

COMPUTER_TOOL: BetaToolComputerUse20250124Param =  {"type": "computer_20250124", "name": "computer", "display_width_px": 1024, "display_height_px": 768}

class ClaudeAgent(Agent):
    def __init__(self, anthropic: Anthropic):
        self.anthropic = anthropic
        self.adapter = ClaudeAdapter()
        self.messages: List[BetaMessageParam] = []
        self.max_iterations = 10
        self.max_tokens = 4096
        self.pending_computer_use_tool_id = None

    async def predict(self, observation: Observation) -> tuple[list[CLA], bool]:
        """
        Predict the next action based on the observation.
        
        Returns:
            tuple[list[CLA], bool]: A tuple containing the list of actions and a boolean indicating if the agent believes it has completed the task.
        """

        # this is the new content that will be sent to the API
        user_content: List[BetaImageBlockParam|BetaTextBlockParam|BetaToolResultBlockParam] = []

        if observation.text:
            logger.info("Adding text to user content: %s", observation.text)
            user_content.append(text_to_content_block(str(observation.text)))
        
        if observation.screenshot:
            logger.info("Adding screenshot to user content")
            if not self.pending_computer_use_tool_id:
                logger.info("Adding screenshot to user content, no tool id")
                user_content.append(base64_to_content_block(observation.screenshot))
            else:
                logger.info("Adding screenshot to user content, tool id: %s", self.pending_computer_use_tool_id)
                user_content.append(
                    tool_use_content_block(
                        self.pending_computer_use_tool_id, 
                        [base64_to_content_block(observation.screenshot)]
                    )
                )
                self.pending_computer_use_tool_id = None

        # Add the user content to the messages
        self.messages.append(cast(BetaMessageParam, {
            "role": "user",
            "content": user_content,
        }))

        # Call Claude API
        response = self.anthropic.beta.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=self.max_tokens,
            messages=self.messages,
            tools=[COMPUTER_TOOL],
            betas=["computer-use-2025-01-24"],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True}
        )

        # Add Claude's response to the conversation history
        response_content = response.content
        self.messages.append(cast(BetaMessageParam, {
            "role": "assistant",
            "content": response_content,
        }))

        # Process tool use
        actions: List[CLA] = []
        for block in response_content:
            logger.info("Processing block: %s", block)
            if block.type == "tool_use":
                logger.info("Processing tool use: %s", block)
                assert block.name == "computer"
                actions.append(self.adapter.convert(block.input))
                self.pending_computer_use_tool_id = block.id
                break

        # If no tools were used, we're done
        done = False
        if not self.pending_computer_use_tool_id:
            logger.info("No more tools used, task completed")
            done = True

        return actions, done
