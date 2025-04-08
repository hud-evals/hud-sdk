"""Information functions for the Text QA environment."""
from __future__ import annotations

import socket
from typing import Any

from qa_controller.utils.state import load_state, save_state


def get_urls() -> dict[str, Any]:
    """Get URLs for this environment.
    
    Returns:
        dict: Dictionary of URLs (empty for this environment)
    """
    # This environment doesn't have a web interface
    return {}


def get_host_ip() -> str:
    """Get the host IP address for the container.
    
    Returns:
        str: IP address
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a public DNS server to determine our IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_state() -> dict[str, Any]:
    """Get the current state of the environment.
    
    Returns:
        dict: Current state information including question and answers count.
    """
    state = load_state()
    answers = state.get("answers", [])
    last_answer = answers[-1] if answers else ""
    
    return {
        "question": state.get("question", ""),
        "answers_count": len(answers),
        "has_answer": bool(answers),
        "last_answer": last_answer
    }


def set_question(question: str) -> dict[str, Any]:
    """Set the current question and clear answers.
    
    Args:
        question: The question text to set
        
    Returns:
        dict: Status of the operation
    """
    state = {"question": question, "answers": []}
    save_state(state)
    return {"status": "success", "question": question}


def reset() -> dict[str, Any]:
    """Reset the environment state (clears question and answers).
    
    Returns:
        dict: Status of the operation
    """
    save_state({"question": "", "answers": []})
    return {"status": "success", "message": "Environment reset"}
