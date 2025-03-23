from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Literal

from textdistance import levenshtein

from hud.evaluators.base import EvaluationResult, Evaluator


class Match(Evaluator):
    """Evaluator that matches responses against expected answers with different matching modes."""

    def __init__(
        self,
        answer: int | str | list[str],
        mode: Literal["auto", "single", "all", "fuzz", "regex", "diff"] = "auto"
    ) -> None:
        """Initialize a Match evaluator.
        
        Args:
            answer: The expected answer(s) to match against
            mode: The matching mode to use:
                - "auto": Automatically choose based on answer type
                - "single": Exact match against a single answer
                - "all": All expected answers must be present
                - "fuzz": Fuzzy string matching
                - "regex": Regular expression matching
                - "diff": Numeric difference comparison or string diff
        """
        self.answer = answer
        self.mode = mode

    def evaluate(self, response: Any) -> EvaluationResult:
        """Evaluate if the response matches the expected answer.
        
        Args:
            response: The response to evaluate
            
        Returns:
            EvaluationResult containing:
            - score: float between 0 and 1
            - reason: explanation of the evaluation
            - mode: mode used for matching
        """
        mode = self._determine_mode() if self.mode == "auto" else self.mode
            
        # Convert response to string if needed
        if not isinstance(response, (str, int, float, list)) and hasattr(response, "__str__"):
            response = str(response)
        
        # Initialize result dictionary
        result: dict[str, Any] = {
            "mode": mode,
        }
        
        if mode == "single":
            passed = self._match_single(response)
            result["score"] = 1.0 if passed else 0.0
            result["reason"] = "Exact match" if passed else "No exact match found"
        
        elif mode == "all":
            answers = [self.answer] if not isinstance(self.answer, list) else self.answer
                
            matches = self._match_all(response, answers)
            result["score"] = matches / len(answers) if answers else 0.0
            
            if matches == len(answers):
                result["reason"] = f"All {matches} expected items found"
            else:
                result["reason"] = f"Only {matches} of {len(answers)} expected items found"
        
        elif mode == "fuzz":
            score = self._match_fuzzy(response, self.answer)
            result["score"] = score
            result["reason"] = f"Fuzzy match with {score:.1%} similarity"
        
        elif mode == "regex":
            passed = self._match_regex(response)
            result["score"] = 1.0 if passed else 0.0
            result["reason"] = "Regex pattern matched" if passed else "Regex pattern did not match"
        
        elif mode == "diff":
            if isinstance(response, (int, float)) and isinstance(self.answer, (int, float)):
                diff_score = self._match_numeric_diff(response, self.answer)
                result["reason"] = f"Numeric difference: {abs(response - self.answer)}"
            else:
                diff_score = self._match_string_diff(response, self.answer)
                result["reason"] = f"String difference with {diff_score:.1%} similarity"
                
            result["score"] = diff_score
        
        # Return EvaluationResult object
        return EvaluationResult(**result)
    
    def _determine_mode(self) -> str:
        """Determine the matching mode based on the answer type."""
        if isinstance(self.answer, (int, float)):
            return "diff"
        elif isinstance(self.answer, str):
            if (self.answer.startswith(("^", ".*")) and
                (self.answer.endswith("$") or self.answer.endswith(".*"))):
                return "regex"
            return "single"
        elif isinstance(self.answer, list):
            return "all"
        return "single"
    
    def _match_single(self, response: Any) -> bool:
        """Check if the answer is present within the response."""
        return str(self.answer).lower().strip() in str(response).lower().strip()
    
    def _match_all(self, response: Any, answers: list) -> int:
        """Count how many expected answers are in the response."""
        response_str = str(response).lower()
        matches = 0
        
        for answer in answers:
            if str(answer).lower() in response_str:
                matches += 1
                
        return matches
    
    def _match_fuzzy(self, response: Any, answer: Any) -> float:
        """Calculate similarity using Levenshtein distance.
        
        Returns a value between 0 and 1, where 1 means identical.
        """
        s1 = str(response).lower()
        s2 = str(answer).lower()
        
        if s1 == s2:
            return 1.0
            
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        # Use Levenshtein distance
        distance = levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)
    
    def _match_regex(self, response: Any) -> bool:
        """Check if response matches regex pattern."""
        try:
            pattern = re.compile(str(self.answer), re.DOTALL)
            return bool(pattern.search(str(response)))
        except re.error:
            return False
    
    def _match_string_diff(self, response: Any, answer: Any) -> float:
        """Compare difference between response and answer strings."""
        matcher = SequenceMatcher(None, str(response), str(answer))
        return matcher.ratio()
        
    def _match_numeric_diff(self, response: float, answer: float) -> float:
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
