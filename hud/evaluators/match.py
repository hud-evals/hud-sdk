from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from textdistance import levenshtein

from hud.evaluators.base import EvaluationResult


def match_single(response: Any, answer: Any) -> EvaluationResult:
    """Check if the answer is present within the response.
    
    Args:
        response: The response to evaluate
        answer: The expected answer
        
    Returns:
        EvaluationResult with score=1.0 if match, 0.0 otherwise
    """
    passed = str(answer).lower().strip() in str(response).lower().strip()
    return EvaluationResult(
        score=1.0 if passed else 0.0,
        reason="Exact match" if passed else "No exact match found",
        mode="single"
    )


def match_all(response: Any, answers: list) -> EvaluationResult:
    """Count how many expected answers are in the response.
    
    Args:
        response: The response to evaluate
        answers: List of expected answers
        
    Returns:
        EvaluationResult with score=proportion of matches (0.0-1.0)
    """
    response_str = str(response).lower()
    matches = 0
    
    for answer in answers:
        if str(answer).lower() in response_str:
            matches += 1
            
    score = matches / len(answers) if answers else 0.0
    
    if matches == len(answers):
        reason = f"All {matches} expected items found"
    else:
        reason = f"Only {matches} of {len(answers)} expected items found"
        
    return EvaluationResult(
        score=score,
        reason=reason,
        mode="all"
    )


def match_fuzzy(response: Any, answer: Any) -> EvaluationResult:
    """Calculate similarity using Levenshtein distance.
    
    Args:
        response: The response to evaluate
        answer: The expected answer
        
    Returns:
        EvaluationResult with score=similarity (0.0-1.0)
    """
    s1 = str(response).lower()
    s2 = str(answer).lower()
    
    if s1 == s2:
        score = 1.0
    elif len(s1) == 0 or len(s2) == 0:
        score = 0.0
    else:
        # Use Levenshtein distance
        distance = levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        score = 1.0 - (distance / max_len)
        
    return EvaluationResult(
        score=score,
        reason=f"Fuzzy match with {score:.1%} similarity",
        mode="fuzz"
    )


def match_regex(response: Any, pattern: str) -> EvaluationResult:
    """Check if response matches regex pattern.
    
    Args:
        response: The response to evaluate
        pattern: Regular expression pattern to match
        
    Returns:
        EvaluationResult with score=1.0 if match, 0.0 otherwise
    """
    try:
        regex = re.compile(pattern, re.DOTALL)
        passed = bool(regex.search(str(response)))
        return EvaluationResult(
            score=1.0 if passed else 0.0,
            reason="Regex pattern matched" if passed else "Regex pattern did not match",
            mode="regex"
        )
    except re.error:
        return EvaluationResult(
            score=0.0,
            reason="Invalid regex pattern",
            mode="regex"
        )


def match_diff(response: Any, answer: Any) -> EvaluationResult:
    """Compare difference between response and answer.
    
    Args:
        response: The response to evaluate
        answer: The expected answer
        
    Returns:
        EvaluationResult with score=similarity (0.0-1.0)
    """
    if isinstance(response, int | float) and isinstance(answer, int | float):
        score = _match_numeric_diff(response, answer)
        reason = f"Numeric difference: {abs(response - answer)}"
    else:
        score = _match_string_diff(response, answer)
        reason = f"String difference with {score:.1%} similarity"
            
    return EvaluationResult(
        score=score,
        reason=reason,
        mode="diff"
    )


def _match_string_diff(response: Any, answer: Any) -> float:
    """Compare difference between response and answer strings."""
    matcher = SequenceMatcher(None, str(response), str(answer))
    return matcher.ratio()
    

def _match_numeric_diff(response: float, answer: float) -> float:
    """Calculate normalized difference between numeric values.
    
    Returns a value between 0 and 1, where 1 means identical and 0 means maximum difference.
    """
    if response == answer:
        return 1.0
        
    # Simple absolute difference normalized to a 0-1 scale
    diff = abs(response - answer)
    max_val = max(abs(response), abs(answer))
    
    if max_val == 0:
        return 1.0  # Both are zero
        
    # Normalize and invert so 1.0 means identical
    return max(0.0, 1.0 - min(1.0, diff / max_val))
