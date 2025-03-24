from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Result of an evaluation.
    
    Attributes:
        score: Float score between 0 and 1
        reason: Explanation of the evaluation
        mode: Mode used for matching, if applicable
    """
    
    score: float
    reason: str
    mode: str | None = None
    criteria_scores: dict[str, float] | None = Field(default_factory=dict)


class Evaluator(ABC):
    """Base class for evaluators that assess task responses."""

    @abstractmethod
    def evaluate(self, response: Any) -> EvaluationResult:
        """Evaluate a response and return an evaluation result.
        
        Args:
            response: The response to evaluate
            
        Returns:
            EvaluationResult containing:
            - score: float between 0 and 1
            - reason: explanation of the evaluation
            - mode: mode used for matching (optional)
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert the evaluator to a dictionary."""


class Passthrough(Evaluator):
    """Evaluator that passes through the response as is."""

    def evaluate(self, response: Any) -> EvaluationResult:
        if isinstance(response, (int, float)):
            score = min(max(float(response), 0.0), 1.0)
        elif isinstance(response, str):
            try:
                score = min(max(float(response), 0.0), 1.0)
            except ValueError:
                score = -1.0
        elif isinstance(response, dict):
            if "reward" in response:
                score = response["reward"]
            elif "score" in response:
                score = response["score"]
            else:
                score = -1.0
        else:
            score = -1.0
        return EvaluationResult(
            score=score,
            reason="Evaluated at environment level"
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "Passthrough",
        }

