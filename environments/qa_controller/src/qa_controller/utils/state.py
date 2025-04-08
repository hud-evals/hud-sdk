"""State management utilities for the Text QA environment."""
from __future__ import annotations

import json
import os
from typing import Any

STATE_FILE = "/tmp/state/qa_state.json"


def save_state(state: dict[str, Any]) -> None:
    """Save environment state to file.
    
    Args:
        state: Dictionary of state data to save
    """
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
        
        
def load_state() -> dict[str, Any]:
    """Load environment state from file.
    
    Returns:
        dict: The current state dictionary, or empty state if none exists
    """
    if not os.path.exists(STATE_FILE):
        return {"question": "", "answers": []}
        
    with open(STATE_FILE) as f:
        return json.load(f)


def get_last_answer() -> str:
    """Get the most recent answer submitted, if any.
    
    Returns:
        str: The last answer, or empty string if none
    """
    state = load_state()
    answers = state.get("answers", [])
    return answers[-1] if answers else ""
