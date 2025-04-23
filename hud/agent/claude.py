import logging
import os
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types.beta import (
    BetaMessageParam,
    BetaToolResultBlockParam,
    BetaToolComputerUse20250124Param,
    BetaTextBlockParam,
    BetaImageBlockParam,
)

from hud.adapters import Adapter
from hud.agent.base import Agent
from hud.adapters.claude import ClaudeAdapter
from hud.env.environment import Observation
from hud.settings import settings

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

# Claude's Computer Use Tool definition
COMPUTER_TOOL: BetaToolComputerUse20250124Param = {
    "type": "computer_20250124", 
    "name": "computer", 
    "display_width_px": 1024, 
    "display_height_px": 768
}

class ClaudeAgent(Agent[AsyncAnthropic, Any]):
    """
    An agent implementation using Anthropic's Claude API with Computer Use.
    
    This agent interacts with HUD environments using Claude's Computer Use API
    through the ClaudeAdapter which converts actions to the format expected by HUD.
    """
    
    def __init__(
        self, 
        client: AsyncAnthropic | None = None,
        adapter: Adapter | None = None,
        model: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 4096,
        max_iterations: int = 10,
    ):
        """
        Initialize the ClaudeAgent.
        
        Args:
            client: The AsyncAnthropic client for API calls (optional, created automatically if not provided)
            adapter: The adapter to use for preprocessing and postprocessing
            model: The Claude model to use
            max_tokens: Maximum tokens for Claude's response
            max_iterations: Maximum number of iterations for the agent
        """
        # Initialize client if not provided
        if client is None:
            # Get API key from settings
            api_key = settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not found in settings or environment variables. Set ANTHROPIC_API_KEY.")
            
            # Create client
            client = AsyncAnthropic(api_key=api_key)

        adapter = adapter or ClaudeAdapter()
            
        super().__init__(client=client, adapter=adapter)
        
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        
        # Default dimensions - will be updated if adapter is provided
        self.width_px = 1024
        self.height_px = 768
        
        # Update dimensions if adapter is provided
        if self.adapter:
            self.width_px = self.adapter.agent_width
            self.height_px = self.adapter.agent_height
        
        # Message history
        self.messages: list[BetaMessageParam] = []
        self.pending_computer_use_tool_id = None

    async def fetch_response(self, observation: Observation) -> tuple[list[Any], bool]:
        """
        Fetch a response from Claude based on the observation.
        
        Args:
            observation: The preprocessed observation
            
        Returns:
            tuple[list[Any], bool]: A tuple containing the list of raw actions and a
                                   boolean indicating if the agent believes the task is complete
        """
        if not self.client:
            raise ValueError("Client is required")
            
        # Prepare the user content for Claude
        user_content: list[BetaImageBlockParam | BetaTextBlockParam | BetaToolResultBlockParam] = []

        # Add text instruction if present
        if observation.text:
            logger.info("Adding text to user content: %s", observation.text)
            user_content.append(text_to_content_block(str(observation.text)))
        
        # Add screenshot if present
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

        # Call Claude API using async client
        response = await self.client.beta.messages.create(
            model=self.model,
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
        actions: list[Any] = []
        done = True  # Assume we're done unless we find a tool use
        
        for block in response_content:
            logger.info("Processing block: %s", block)
            if block.type == "tool_use":
                logger.info("Processing tool use: %s", block)
                assert block.name == "computer"
                
                # Store the raw action
                actions.append(block.input)
                self.pending_computer_use_tool_id = block.id
                
                # If we found a tool use, we're not done
                done = False
                break

        # If no tool use action was found, check for a final text response
        if not actions and done:
            final_text_response = ""
            for block in response_content:
                if block.type == "text":
                    final_text_response += block.text
            
            if final_text_response.strip():
                logger.info(f"No tool use found. Using final text as response: {final_text_response}")
                actions = [{
                    "action": "response", 
                    "text": final_text_response.strip()
                }]
                # Keep done = True
            else:
                 logger.info("No tool use and no final text block found.")
                 # Keep done = True, actions remains empty
            
        return actions, done
