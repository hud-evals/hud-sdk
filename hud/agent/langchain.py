import logging
from typing import Any, Generic, List, Optional, TypeVar, Union, cast

# Langchain imports
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableSerializable
from pydantic import Field, BaseModel

# HUD imports
from hud.adapters import Adapter
from hud.agent.base import Agent
from hud.env.environment import Observation
from hud.adapters.common.types import (
    CLA,
    ClickAction,
    TypeAction,
    ScrollAction,
    MoveAction,
    DragAction,
    PressAction,
    KeyDownAction,
    KeyUpAction,
    WaitAction,
    ResponseAction,
    CustomAction,
    # Exclude ScreenshotFetch, PositionFetch as they are internal
)

logger = logging.getLogger(__name__)

# Define a Pydantic Union type representing exactly ONE possible CLA action
# This is what we'll ask the Langchain model to output.
SingleCLAction = Union[
    ClickAction,
    TypeAction,
    ScrollAction,
    MoveAction,
    DragAction,
    PressAction,
    KeyDownAction,
    KeyUpAction,
    WaitAction,
    ResponseAction,
]

# Define a Pydantic model to wrap the single action, potentially making it
# easier for the LLM to consistently output the desired structure.
class StepAction(BaseModel):
    """Wrapper model requesting a single concrete CLA action from the Langchain model."""
    action: SingleCLAction = Field(..., description="The single CLA action to perform for this step.")

# Generic Type for the Langchain Model/Runnable
# Allows flexibility in what the user provides (model, chain, etc.)
# Bound to BaseLanguageModel as .with_structured_output is expected
LangchainModelOrRunnable = TypeVar("LangchainModelOrRunnable", bound=BaseLanguageModel)

class LangchainAgent(Agent[LangchainModelOrRunnable, Any], Generic[LangchainModelOrRunnable]):
    """
    An agent that uses an arbitrary Langchain model or runnable, leveraging
    Langchain's structured output capabilities to produce a single CLA action per step.
    """

    def __init__(
        self,
        langchain_model: LangchainModelOrRunnable,
        adapter: Optional[Adapter] = None,
        system_prompt: str | None = None,
    ):
        """
        Initialize the LangchainAgent.

        Args:
            langchain_model: The Langchain language model or runnable chain to use.
                             Must support asynchronous invocation (`ainvoke`) and
                             `.with_structured_output()`.
            adapter: An optional HUD adapter. If provided, it will be used for
                     preprocessing observations (rescaling) and postprocessing
                     the single CLA action (coordinate rescaling).
            system_prompt: An optional system prompt to guide the Langchain model.
                           If None, a default prompt encouraging single CLA output is used.
        """
        super().__init__(client=langchain_model, adapter=adapter) # Store model as 'client'
        self.langchain_model = langchain_model # Also store with specific name

        self.system_prompt_str = system_prompt or self._get_default_system_prompt()
        self.history: List[BaseMessage] = []

    def _get_default_system_prompt(self) -> str:
        # TODO: Refine this prompt based on testing.
        # It needs to strongly encourage outputting *only* the StepAction structure.
        return (
            "You are an agent interacting with a computer environment (either a web browser or an OS desktop). "
            "Your goal is to follow the user's instructions based on the provided text and screenshot observations."
            "For each step, you must choose exactly ONE action to perform from the available CLA action types."
            "Output your chosen action using the provided 'StepAction' tool/function."
            "If you believe the task is complete based on the user's prompt and the observations, use the 'ResponseAction'."
        )

    async def fetch_response(self, observation: Observation) -> tuple[CLA | None, bool]:
        """
        Fetches a response from the configured Langchain model, expecting a single
        structured CLA action.

        Args:
            observation: The preprocessed observation (screenshot potentially rescaled by adapter).

        Returns:
            A tuple containing:
            - A list with a single dictionary representing the raw CLA action (before adapter postprocessing).
            - A boolean indicating if the agent chose ResponseAction (task completion).
        """
        # 1. Format observation into Langchain message(s)
        human_content: List[Union[str, dict]] = []
        if observation.text:
            human_content.append(observation.text)
        if observation.screenshot:
            # Assuming the Langchain model/chain can handle base64 images
            # This might need adjustment based on the specific model used.
            human_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{observation.screenshot}"
                }
            })
        
        if not human_content:
             logger.warning("LangchainAgent received an observation with no text or screenshot.")
             # Decide how to handle empty observation - perhaps return no action?
             return [], False # Or raise an error?

        current_human_message = HumanMessage(content=human_content)

        # 2. Prepare message history for the model
        messages_for_llm: List[BaseMessage] = [
            SystemMessage(content=self.system_prompt_str),
            *self.history,
            current_human_message,
        ]

        # 3. Configure structured output
        # We ask for the StepAction wrapper, which contains the actual SingleCLAAction
        # Explicitly use method="function_calling" to handle schemas with default values
        structured_llm = self.langchain_model.with_structured_output(
            schema=StepAction, 
            method="function_calling"
        )

        # 4. Invoke Langchain model asynchronously
        try:
            ai_response_structured = await structured_llm.ainvoke(messages_for_llm)
        except Exception as e:
            logger.error(f"Langchain model invocation failed: {e}", exc_info=True)
            # Decide how to handle LLM errors - maybe retry or return empty action?
            return [], False

        # 5. Process the structured response
        is_done = False
        ai_message_content_for_history = "" # For storing in history

        if isinstance(ai_response_structured, StepAction):
            # Successfully got the wrapper, extract the actual action
            actual_action = ai_response_structured.action
            ai_message_content_for_history = actual_action.model_dump()
            if isinstance(actual_action, ResponseAction):
                is_done = True
                logger.info(f"LangchainAgent determined task is done with response: {actual_action.text[:100]}...")
            else:
                 logger.info(f"LangchainAgent produced action: {type(actual_action).__name__}")

        else:
            logger.warning(
                f"Langchain model did not return the expected StepAction structure. "
                f"Received type: {type(ai_response_structured)}. Value: {ai_response_structured!r}"
            )
             # Attempt to add raw response to history for debugging
            if isinstance(ai_response_structured, BaseMessage):
                 ai_message_content_for_history = ai_response_structured.content
            elif isinstance(ai_response_structured, str):
                 ai_message_content_for_history = ai_response_structured
            else:
                 ai_message_content_for_history = repr(ai_response_structured)
            # Return no action as we didn't get the expected structure
            return [], False

        # 6. Update history
        self.history.append(current_human_message)
        # Add the AI response (containing the structured action dict) to history
        # Convert dict to string representation for AIMessage content
        self.history.append(AIMessage(content=repr(ai_message_content_for_history)))
        # TODO: Consider history truncation/summarization if it grows too long

        if actual_action:
            # Return the single action dictionary within a list
            return [actual_action], is_done
        else:
            # Should ideally not happen if structure validation worked, but as a fallback
            return [], is_done 