from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.evaluators.base import EvaluationResult, Evaluator


class Task:
    """A task that can be executed and evaluated.
    
    Attributes:
        prompt: The task prompt or instruction
        setup: Environment setup configuration (optional)
        extract: Configuration for extracting response from the environment (optional)
        evaluator: Evaluator for assessing responses (optional)
    """
    
    def __init__(
        self,
        prompt: str,
        setup: dict[str, Any] | None = None,
        extract: dict[str, Any] | None = None,
        evaluator: Evaluator | None = None
    ) -> None:
        """Initialize a task.
        
        Args:
            prompt: The task prompt or instruction
            setup: Environment setup configuration
            extract: Optional configuration for extracting response
            evaluator: Evaluator for assessing responses
        """
        self.prompt = prompt
        self.setup = setup
        self.extract = extract
        self.evaluator = evaluator or Evaluator()
    
    def evaluate(self, response: Any) -> EvaluationResult:
        """Evaluate a response using the task's evaluator.
        
        Args:
            response: The response to evaluate
            
        Returns:
            EvaluationResult with score and reason
        """
        if not self.evaluator:
            raise ValueError("No evaluator defined for this task")
            
        return self.evaluator.evaluate(response)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create a Task from a dictionary.
        
        Args:
            data: Dictionary containing task data
            
        Returns:
            Task instance
        """
        # Note: The evaluator needs to be created separately and passed in
        return cls(
            prompt=data["prompt"],
            setup=data.get("setup", {}),
            extract=data.get("extract"),
            evaluator=data.get("evaluator")
        )
        
    def to_dict(self) -> dict[str, Any]:
        """Convert Task to a dictionary.
        
        Returns:
            Dictionary representation of the task
        """
        result = {
            "prompt": self.prompt,
            "setup": self.setup,
        }
        
        if self.extract:
            result["extract"] = self.extract
            
        return result
