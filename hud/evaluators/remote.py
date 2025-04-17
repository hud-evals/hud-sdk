from __future__ import annotations

import asyncio
from typing import Any

from hud.evaluators.base import EvaluationResult
from hud.server import make_request
from hud.settings import settings


async def _remote_eval_call(
    response: Any,
    answer: Any,
    eval_type: str,
    config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Send an evaluation request to the remote server.
    
    Args:
        response: The response to evaluate
        answer: The reference answer to compare against
        eval_type: Type of evaluation (e.g., "match", "judge", "agent")
        config: Optional configuration parameters
        
    Returns:
        Dictionary with evaluation results from the server
    """
    try:
        result = await make_request(
            method="POST",
            url=f"{settings.base_url}/evaluations/evaluate",
            json={
                "response": response,
                "answer": answer,
                "type": eval_type,
                "config": config or {}
            },
            api_key=settings.api_key,
        )
        return result
    except Exception as e:
        return {
            "score": -1.0,
            "reason": f"Remote evaluation failed: {e!s}",
            "details": {}
        }


def remote_evaluate(
    response: Any,
    answer: Any,
    eval_type: str = "default",
    config: dict[str, Any] | None = None
) -> EvaluationResult:
    """Evaluate a response using remote evaluation services.
    
    Args:
        response: The response to evaluate
        answer: The reference answer to compare against
        eval_type: Type of evaluation to perform
        config: Optional configuration for the evaluation
        
    Returns:
        EvaluationResult containing the evaluation results
    """
    result = asyncio.run(_remote_eval_call(
        response=response,
        answer=answer,
        eval_type=eval_type,
        config=config
    ))
    
    return EvaluationResult(
        score=result.get("score", -1.0),
        reason=result.get("reason", "Remote evaluation completed"),
        mode=eval_type,
        criteria_scores=result.get("details", {})
    )
