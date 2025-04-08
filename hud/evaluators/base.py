from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hud.task import Task


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
    """Abstract base class for evaluators."""
    
    @abstractmethod
    def evaluate(self, task: Task, response: str) -> EvaluationResult:
        """Evaluate a task and response."""
