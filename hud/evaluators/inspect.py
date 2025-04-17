from __future__ import annotations

from typing import Any

from hud.evaluators.base import EvaluationResult


def inspect_evaluate(
    response: Any,
    answer: Any,
) -> EvaluationResult:
    """Evaluate using Inspect-ai's evaluation models.
    
    Args:
        response: The response to evaluate
        answer: The reference answer to compare against
        model_name: The Inspect model to use
        prompt: Optional custom prompt for evaluation
        metrics: Optional list of metrics to evaluate against
        
    Returns:
        EvaluationResult with the evaluation results
    """
    return EvaluationResult(
        score=0.0,
        reason="Inspect evaluation not implemented",
        mode="inspect"
    )

