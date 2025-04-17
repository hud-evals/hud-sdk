"""Text matching evaluation functions."""
from __future__ import annotations

from typing import Any

from qa_controller.utils.state import get_last_answer

# Try to import nltk for better evaluation
try:
    import nltk
    # Check if nltk data is available, download if needed
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def exact_match(reference: str) -> dict[str, Any]:
    """Check if the last answer exactly matches the reference.
    
    Args:
        reference: The reference answer to check against
        
    Returns:
        dict: Evaluation result with score and explanation
    """
    last_answer = get_last_answer()
    
    if not last_answer:
        return {
            "score": 0.0,
            "reason": "No answer submitted yet"
        }
    
    is_match = last_answer.lower().strip() == reference.lower().strip()
    
    return {
        "score": 1.0 if is_match else 0.0,
        "reason": "Exact match" if is_match else "No exact match",
        "submitted": last_answer
    }


def fuzzy_match(reference: str) -> dict[str, Any]:
    """Check similarity between last answer and reference.
    
    Args:
        reference: The reference answer to check against
        
    Returns:
        dict: Evaluation result with similarity score
    """
    last_answer = get_last_answer()
    
    if not last_answer:
        return {
            "score": 0.0,
            "reason": "No answer submitted yet"
        }
    
    last_answer_lower = last_answer.lower()
    reference_lower = reference.lower()
    
    if HAS_NLTK:
        # Use NLTK for token-based comparison
        answer_tokens = set(nltk.word_tokenize(last_answer_lower))
        reference_tokens = set(nltk.word_tokenize(reference_lower))
        
        if not reference_tokens:
            return {"score": 0.0, "reason": "Empty reference"}
            
        overlap = len(answer_tokens.intersection(reference_tokens))
        total = len(reference_tokens)
        score = overlap / total if total > 0 else 0.0
        
        return {
            "score": score,
            "reason": f"Token overlap: {overlap}/{total}",
            "submitted": last_answer
        }
    else:
        # Fallback to simple word matching
        answer_words = set(last_answer_lower.split())
        reference_words = set(reference_lower.split())
        
        if not reference_words:
            return {"score": 0.0, "reason": "Empty reference"}
            
        overlap = len(answer_words.intersection(reference_words))
        total = len(reference_words)
        score = overlap / total if total > 0 else 0.0
        
        return {
            "score": score,
            "reason": f"Word overlap: {overlap}/{total}",
            "submitted": last_answer
        }


def contains_keywords(keywords: list[str]) -> dict[str, Any]:
    """Check if the last answer contains specific keywords.
    
    Args:
        keywords: List of keywords to check for
        
    Returns:
        dict: Evaluation result with match details
    """
    last_answer = get_last_answer()
    
    if not last_answer:
        return {
            "score": 0.0,
            "reason": "No answer submitted yet"
        }
    
    last_answer_lower = last_answer.lower()
    
    # Check how many keywords are present
    matches = []
    for keyword in keywords:
        if keyword.lower() in last_answer_lower:
            matches.append(keyword)
    
    score = len(matches) / len(keywords) if keywords else 0.0
    
    return {
        "score": score,
        "reason": f"Found {len(matches)}/{len(keywords)} keywords",
        "matches": matches,
        "submitted": last_answer
    }
