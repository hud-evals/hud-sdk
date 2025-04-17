from __future__ import annotations

import asyncio
import base64
from typing import Any, Protocol, TypedDict

from hud.evaluators.base import EvaluationResult
from hud.server import make_request
from hud.settings import settings


class LLM(Protocol):
    """Protocol for LLM interfaces that can be used for evaluation."""
    async def ainvoke(self, prompt: str) -> str: ...


class Criterion(TypedDict, total=False):
    """Criterion for judge-based evaluation."""
    
    description: str
    weight: float


async def _call_eval_endpoint(
    response: Any,
    answer: Any,
    criteria: list[Any],
    mode: str
) -> dict[str, Any]:
    """Call the run_eval endpoint to evaluate the response."""
    try:
        result = await make_request(
            method="POST",
            url=f"{settings.base_url}/evaluations/run_eval",
            json={
                "response": response,
                "answer": answer,
                "criteria": criteria,
                "mode": mode
            },
            api_key=settings.api_key,
        )
        return result
    except Exception as e:
        # Fallback to local evaluation if remote call fails
        return {
            "score": -1.0,
            "reason": f"Remote evaluation failed: {e!s}. Fallback to default score.",
            "criteria_scores": {}
        }


def _determine_mode(answer: Any) -> str:
    """Determine the evaluation mode based on answer type."""
    if isinstance(answer, bytes) or _is_base64_image(answer):
        return "VLM"
    return "LLM"


def _process_input(data: Any) -> Any:
    """Process input data, detecting and handling base64 images."""
    if isinstance(data, bytes):
        # Convert bytes to base64 string
        return base64.b64encode(data).decode("utf-8")
    
    if isinstance(data, str) and _is_base64_image(data):
        # It's already a base64 string, just return it
        return data
        
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        # Process list of strings
        return data
        
    # For other types, convert to string
    return str(data) if not isinstance(data, str | dict) else data


def _is_base64_image(data: Any) -> bool:
    """Check if a string is a base64 encoded image."""
    if not isinstance(data, str):
        return False
        
    # Check for common image data URI pattern
    if data.startswith(("data:image/", "data:application/octet-stream")):
        return True
        
    # Check if it's a base64 encoded string with image header
    try:
        # First, validate it's base64 decodable
        padding_needed = len(data) % 4
        if padding_needed:
            data += "=" * (4 - padding_needed)

        # Try to decode the first few bytes to check for image signatures
        sample = base64.b64decode(data[:30])

        # Check for common image format signatures
        return (
            sample.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"RIFF"))
        )
    except Exception:
        return False


def judge(
    response: Any,
    answer: Any,
    llm: LLM | None = None,
    criteria: list[str] | list[dict] | None = None,
) -> EvaluationResult:
    """Judge a response against an answer using an LLM.
    
    Args:
        response: The response to evaluate
        answer: The reference answer to compare against
        llm: Optional langchain LLM to use for evaluation
        criteria: Evaluation criteria as strings or dictionaries
        
    Returns:
        EvaluationResult with evaluation results
    """
    # Process inputs
    processed_response = _process_input(response)
    processed_answer = _process_input(answer)
    
    # If LLM is provided, use it for evaluation
    if llm:
        return _evaluate_with_llm(processed_response, processed_answer, llm, criteria)
    
    # Otherwise, use the remote evaluation service
    mode = "LLM"
    if isinstance(answer, bytes) or _is_base64_image(answer):
        mode = "VLM"
    
    # Call the eval endpoint synchronously
    result = asyncio.run(_call_eval_endpoint(
        response=processed_response,
        answer=processed_answer,
        criteria=criteria or [],
        mode=mode
    ))
    
    return EvaluationResult(
        score=result.get("score", -1.0),
        reason=result.get("reason", "Response evaluated"),
        mode=mode,
        criteria_scores=result.get("criteria_scores", {})
    )


def _evaluate_with_llm(
    response: Any,
    answer: Any,
    llm: LLM,
    criteria: list[str] | list[dict] | None = None
) -> EvaluationResult:
    """Evaluate a response against an answer using a provided LLM."""
    criteria_text = ""
    if criteria:
        criteria_text = "Use the following criteria:\n"
        for c in criteria:
            if isinstance(c, dict) and "description" in c:
                criteria_text += f"- {c['description']}\n"
            elif isinstance(c, str):
                criteria_text += f"- {c}\n"
    
    prompt = f"""Evaluate the quality of a response given a reference answer.

REFERENCE ANSWER:
{answer}

RESPONSE TO EVALUATE:
{response}

{criteria_text}
Rate the response on a scale from 0.0 to 1.0, where 1.0 is perfect.
Provide a brief explanation for your rating.
Format your answer as a JSON object with 'score' (float) and 'reason' (string) fields.
"""

    try:
        # Run the evaluation asynchronously
        result_text = asyncio.run(llm.ainvoke(prompt))
        
        # Attempt to parse JSON response
        import json
        import re
        
        # Try to extract JSON if wrapped in other text
        json_match = re.search(r"\{.*?\}", result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            return EvaluationResult(
                score=float(result.get("score", 0.5)),
                reason=result.get("reason", "Evaluated with custom LLM"),
                mode="custom_llm"
            )
        
        # If can't parse as JSON, use default values
        return EvaluationResult(
            score=0.5,
            reason=f"Unable to parse LLM response as JSON. Raw response: {result_text[:100]}...",
            mode="custom_llm"
        )
        
    except Exception as e:
        return EvaluationResult(
            score=0.0,
            reason=f"LLM evaluation error: {e!s}",
            mode="custom_llm"
        )
