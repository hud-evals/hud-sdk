"""Text matching evaluation functions."""
from __future__ import annotations

from typing import Any

from hud_controller.utils.state import get_last_answer


def exact_match(reference: str) -> dict[str, Any]:
    """Check if the last answer exactly matches the reference.
    
    Args:
        reference: The reference answer to check against
        
    Returns:
        dict: Evaluation result with reward and explanation
    """
    last_answer = get_last_answer()
    
    if not last_answer:
        return {
            "reward": 0.0,
            "reason": "No answer submitted yet"
        }
    
    is_match = last_answer.lower().strip() == reference.lower().strip()
    
    return {
        "reward": 1.0 if is_match else 0.0,
        "reason": "Exact match" if is_match else "No exact match",
        "submitted": last_answer
    }


def fuzzy_match(reference: str) -> dict[str, Any]:
    """Check similarity between last answer and reference.
    
    Args:
        reference: The reference answer to check against
        
    Returns:
        dict: Evaluation result with similarity reward
    """
    last_answer = get_last_answer()
    
    if not last_answer:
        return {
            "reward": 0.0,
            "reason": "No answer submitted yet"
        }
    
    last_answer_lower = last_answer.lower()
    reference_lower = reference.lower()
    
    # Simple word matching
    answer_words = set(last_answer_lower.split())
    reference_words = set(reference_lower.split())
    
    if not reference_words:
        return {"reward": 0.0, "reason": "Empty reference"}
        
    overlap = len(answer_words.intersection(reference_words))
    total = len(reference_words)
    reward = overlap / total if total > 0 else 0.0
    
    return {
        "reward": reward,
        "reason": f"Word overlap: {overlap}/{total}",
        "submitted": last_answer
    }


def contains_keywords(keywords: str | list[str]) -> dict[str, Any]:
    """Check if the last answer contains specified keywords.
    
    Args:
        keywords: Single keyword or list of keywords to check for
        
    Returns:
        dict: Evaluation result with reward and explanation
    """
    last_answer = get_last_answer()
    
    if not last_answer:
        return {
            "reward": 0.0,
            "reason": "No answer submitted yet"
        }
    
    # Convert single keyword to list
    if isinstance(keywords, str):
        keywords = [keywords]
    
    # Check if answer contains any of the keywords
    last_answer_lower = last_answer.lower()
    found_keywords = []
    
    for keyword in keywords:
        if keyword.lower() in last_answer_lower:
            found_keywords.append(keyword)
    
    has_keywords = len(found_keywords) > 0
    
    return {
        "reward": 1.0 if has_keywords else 0.0,
        "reason": f"Found keywords: {found_keywords}" if has_keywords else "No keywords found",
        "submitted": last_answer,
        "found": found_keywords
    }
