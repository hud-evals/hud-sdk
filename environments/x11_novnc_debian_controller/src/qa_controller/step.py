"""Step function for the Text QA environment."""
from __future__ import annotations

from typing import Any

from qa_controller.utils.state import load_state, save_state


def step(command: str) -> dict[str, Any]:
    """Execute a step in the environment.
    
    Args:
        command: The command to execute, expected to be "answer: <text>"
        
    Returns:
        dict: Result of the step execution
    """
    # Parse command - expecting "answer: <text>"
    if isinstance(command, str) and command.startswith("answer:"):
        answer = command[7:].strip()
        state = load_state()
        
        # Check if there's a question
        if not state.get("question"):
            return {
                "status": "error",
                "message": "No question has been set"
            }
        
        # Add answer to history
        if "answers" not in state:
            state["answers"] = []
        state["answers"].append(answer)
        
        # Save updated state
        save_state(state)
        return {
            "status": "success",
            "answer_submitted": answer,
            "question": state["question"]
        }
    else:
        return {
            "status": "error",
            "message": "Unknown command. Use 'answer: <your answer>'"
        }
