import os
import json
from typing import Union, Optional, Tuple, List, Dict, Any
from agent.base import Agent
from anthropic import AsyncAnthropic  # Import AsyncAnthropic
from anthropic.types import Message


class AsyncClaudeAgent(Agent):  # Rename class
    def __init__(self, client: AsyncAnthropic):  # Change client type hint
        super().__init__(client)
        self.model = "claude-3-7-sonnet-20250219"
        self.max_tokens = 4096
        self.tool_version = "20250124"
        self.thinking_budget = 1024
        self.conversation: List[Dict[str, Any]] = []  # Store the full conversation history including Claude's responses
        self.responses: List[Message] = [] # Assuming Agent base class has responses attribute, initialize it here if not.
        self.pending_tool_use_id: Optional[str] = None # State to track pending tool results

    async def predict(
        self, screenshot: Optional[str] = None, text: Optional[str] = None
    ) -> Tuple[bool, Union[str, object, None]]: # Fix type hints
        # Create the user message (could be regular or tool_result)
        message = self._create_message(screenshot, text)

        # Only append the message if it's not empty
        if message:
            self.conversation.append(message)
            # Ensure history doesn't grow indefinitely (optional, add if needed)
            # self.conversation = self.conversation[-MAX_HISTORY:]

        # If the last message was a tool_result, we expect an assistant response
        # If the last message was a regular user message, we expect an assistant response
        # If no message was added (e.g., waiting for screenshot for a tool_result),
        # maybe we shouldn't call generate_response?
        # However, the current logic proceeds, let's stick to it unless issues arise.
        # The API call might fail if the sequence is wrong (e.g., tool_use not followed by tool_result)

        try:
            response = await self._generate_response() # Await async call
        except Exception as e:
            # Catch potential API errors due to bad conversation history earlier
            print(f"Error generating response, possibly due to conversation state: {e}")
            # Decide how to handle this - maybe return an error state?
            # For now, re-raise to see the error.
            raise

        # Add Claude's response to the conversation history
        # Ensure response.content structure is handled correctly
        assistant_message = {"role": "assistant", "content": response.content if response.content else []}
        self.conversation.append(assistant_message)

        # Assuming self.responses exists and is meant to store raw responses
        self.responses.append(response)

        # Process the response to check for final text or new tool use requests
        done, processed = await self.process_response(response)

        return done, processed

    def _create_message(self, screenshot: Optional[str] = None, text: Optional[str] = None) -> Optional[Dict[str, Any]]: # Fix type hints
        """Create appropriate message based on context and inputs. Prioritizes sending tool_result if pending."""

        # --- Prioritize Tool Result ---
        if self.pending_tool_use_id:
            tool_use_id_to_respond = self.pending_tool_use_id
            # Clear pending ID *before* returning the message that satisfies it
            self.pending_tool_use_id = None

            content_for_result = []
            if screenshot:
                content_for_result.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot,
                        },
                    }
                )
            # If no screenshot, send an empty content list for the tool_result.
            # Anthropic might require specific content even for errors,
            # review their docs if this causes issues.
            # We are consuming the pending_tool_use_id, so we MUST send a result.

            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id_to_respond,
                        "content": content_for_result,
                        # Note: Anthropic documentation suggests content can be text or image.
                        # If text was also provided, should it be included here?
                        # For now, keeping it simple: only screenshot contributes to tool_result content.
                        # 'is_error=True' could be added if screenshot was required but missing.
                    }
                ],
            }
        # --- End Tool Result Priority ---


        # --- Regular user message (only if no tool_result was pending) ---
        if text or screenshot:
            content: List[Dict[str, Any]] = []
            if text:
                content.append({"type": "text", "text": text})
            if screenshot:
                # This appends a screenshot as a regular image input,
                # not as a response to a tool use.
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot,
                        },
                    }
                )

            return {"role": "user", "content": content}

        # If no pending tool_use_id and no text/screenshot provided, return None.
        return None

    async def _generate_response(self) -> Message: # Make async
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

        # Ensure conversation is not empty before sending
        if not self.conversation:
             raise ValueError("Cannot generate response with empty conversation history.")

        try:
            # Use await with the async client method
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=self.conversation,  # Use the full conversation
                tools=tools,
                betas=[beta_flag],
                thinking=thinking,
            )
            return response
        except Exception as e:
            # Consider more specific error handling/logging
            print(f"Error calling Anthropic API: {e}")
            # Log the conversation history that caused the error for debugging
            # Be careful about logging sensitive data if screenshots are involved
            # print(f"Conversation state: {json.dumps(self.conversation, indent=2)}")
            raise

    async def process_response(
        self, response: Message
    ) -> Tuple[bool, Union[str, object, None]]: # Fix type hints
        """
        Process the assistant's response.
        Sets self.pending_tool_use_id if the response ends with a tool use.
        Clears self.pending_tool_use_id if the response ends with text.
        Returns (is_final, processed_result) where processed_result is
        text if final, or tool input if not final.
        """
        computer_action = None
        tool_use_id = None
        text_response = None
        is_final = True # Assume final unless a tool use is the definitive end
        processed_result: Union[str, object, None] = None

        # Default: clear pending ID unless we find a tool use at the end
        self.pending_tool_use_id = None

        if response.content:
            # Check the *last* block to determine the final state
            last_block = response.content[-1]
            if hasattr(last_block, "type"):
                if last_block.type == "text":
                    is_final = True
                    processed_result = getattr(last_block, 'text', None)
                    # self.pending_tool_use_id remains None (cleared by default)
                elif last_block.type == "tool_use" and hasattr(last_block, "name") and last_block.name == "computer":
                    is_final = False
                    processed_result = getattr(last_block, 'input', None) # The computer action
                    # Set pending ID as we expect a tool_result next
                    self.pending_tool_use_id = getattr(last_block, 'id', None)
                    if not self.pending_tool_use_id:
                         # This shouldn't happen based on API spec, but handle defensively
                         print("Warning: Tool use block received without an ID.")
                         is_final = True # Treat as final if ID is missing
                         processed_result = None
                else:
                    # Unexpected last block type, treat as final with no specific result
                    print(f"Warning: Unexpected last block type: {last_block.type}")
                    is_final = True
                    processed_result = None
                    # self.pending_tool_use_id remains None
            else:
                 # Last block has no type attribute, treat as final
                 is_final = True
                 processed_result = None
                 # self.pending_tool_use_id remains None

        else:
            # No content in response, treat as final
            is_final = True
            processed_result = None
            # self.pending_tool_use_id remains None

        return is_final, processed_result 