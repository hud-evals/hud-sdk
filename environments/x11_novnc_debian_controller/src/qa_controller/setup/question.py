"""Question management functions for the QA environment."""
from __future__ import annotations

from typing import Any

from qa_controller.utils.state import load_state, save_state
    

def set_question(question: str) -> dict[str, Any]:
    """Set the current question.
    
    Args:
        question: The question text to set
        
    Returns:
        dict: Status of the operation
    """
    state = {"question": question, "answers": []}
    save_state(state)
    return {"status": "success", "question": question}
    
    
def reset() -> dict[str, Any]:
    """Reset the environment state.
    
    Returns:
        dict: Status of the operation
    """
    save_state({"question": "", "answers": []})
    return {"status": "success", "message": "Environment reset"}


def get_question() -> dict[str, Any]:
    """Get the current question.
    
    Returns:
        dict: The current question
    """
    state = load_state()
    return {
        "question": state.get("question", ""),
        "answers_count": len(state.get("answers", []))
    }
