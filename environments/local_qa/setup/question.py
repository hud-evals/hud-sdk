"""Question management functions for the QA environment."""

import os
import sys
from typing import Any

# Add the parent directory to sys.path to make relative imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.state import load_state, save_state
    

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