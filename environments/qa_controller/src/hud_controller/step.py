"""Step function for the Text QA environment."""
from __future__ import annotations

from typing import Any

from hud_controller.utils.state import load_state, save_state


def step(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute a step in the environment.
    
    Args:
        actions: List of actions to execute
        
    Returns:
        dict: Result of the step execution with observation format
    """
    # Load current state
    state = load_state()
    
    # If no question is set, set a default one
    if not state.get("question"):
        state["question"] = "What is 2 + 2?"
        save_state(state)
    
    # Process actions if provided
    if actions:
        for action in actions:
            if action.get("type") == "response":
                answer = action.get("text", "")
                
                # Add answer to history
                if "answers" not in state:
                    state["answers"] = []
                state["answers"].append(answer)
                
                # Save updated state
                save_state(state)
                
                # Create observation text
                observation_text = f"Question: {state['question']}\nYour answer '{answer}' has been recorded."
                break
        else:
            # No response action found
            observation_text = f"Question: {state['question']}\nPlease provide your answer using a response action."
    else:
        # No actions provided, show current question
        observation_text = f"Question: {state['question']}\nPlease provide your answer using a response action."
    
    # Return in the format expected by HUD SDK
    return {
        "observation": {
            "screenshot": None,  # No screenshot for text-based QA
            "text": observation_text
        }
    }
