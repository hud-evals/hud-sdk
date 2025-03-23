from __future__ import annotations

import asyncio
import base64
from typing import Any, Literal, TypedDict

from hud.evaluators.base import EvaluationResult, Evaluator
from hud.server import make_request
from hud.settings import settings


class Criterion(TypedDict, total=False):
    """Criterion for judge-based evaluation."""
    
    description: str
    weight: float


class Judge(Evaluator):
    """Evaluator that judges responses based on custom criteria using remote evaluation."""

    def __init__(
        self,
        answer: str | list[str] | bytes,
        criteria: list[str] | list[Criterion] | None = None,
        mode: Literal["auto", "LLM", "VLM"] = "auto"
    ) -> None:
        """Initialize a Judge evaluator.
        
        Args:
            answer: The reference answer to compare against
            criteria: Evaluation criteria as strings or Criterion dictionaries
            mode: The evaluation mode to use:
                - "auto": Automatically choose based on answer type
                - "LLM": Use a language model for evaluation
                - "VLM": Use a vision language model for evaluation
        """
        self.answer = answer
        self.criteria = criteria or []
        self.mode = mode

    def evaluate(self, response: Any) -> EvaluationResult:
        """Judge a response based on the specified criteria by calling the run_eval endpoint.
        
        Args:
            response: The response to evaluate
            
        Returns:
            EvaluationResult containing:
            - score: float between 0 and 1
            - reason: explanation of the evaluation
            - mode: mode used for evaluation
            - criteria_scores: individual scores for each criterion
        """
        mode = self._determine_mode() if self.mode == "auto" else self.mode
        
        # Process response and reference answer
        processed_response = self._process_input(response)
        processed_answer = self._process_input(self.answer)
        
        # Call the eval endpoint synchronously
        result = asyncio.run(self._call_eval_endpoint(
            response=processed_response,
            answer=processed_answer,
            criteria=self.criteria,
            mode=mode
        ))
        
        # Convert to EvaluationResult and return
        return EvaluationResult(
            score=result.get("score", -1.0),
            reason=result.get("reason", "Response evaluated"),
            mode=mode,
            criteria_scores=result.get("criteria_scores", {})
        )
    
    async def _call_eval_endpoint(
        self,
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
                    "criteria": criteria, # TODO backend
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
    
    def _determine_mode(self) -> str:
        """Determine the evaluation mode based on answer type."""
        if isinstance(self.answer, bytes) or self._is_base64_image(self.answer):
            return "VLM"
        return "LLM"
    
    def _process_input(self, data: Any) -> Any:
        """Process input data, detecting and handling base64 images."""
        if isinstance(data, bytes):
            # Convert bytes to base64 string
            return base64.b64encode(data).decode("utf-8")
        
        if isinstance(data, str) and self._is_base64_image(data):
            # It's already a base64 string, just return it
            return data
            
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            # Process list of strings
            return data
            
        # For other types, convert to string
        return str(data) if not isinstance(data, (str, dict)) else data
    
    def _is_base64_image(self, data: Any) -> bool:
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
