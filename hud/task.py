from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud.evaluators import Passthrough, Match, Judge

if TYPE_CHECKING:
    from hud.evaluators.base import EvaluationResult, Evaluator

EVALUATORS = {
    "LLM": Judge,
    "VLM": Judge,
    "Match": Match,
}

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
        config: dict[str, Any] | None = None,
        extract: dict[str, Any] | None = None,
        evaluator: Evaluator | None = None,
        evaluator_config: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize a task.
        
        Args:
            prompt: The task prompt or instruction
            setup: Optional environment setup configuration
            extract: Optional configuration for extracting response
            evaluator: Optional evaluator for assessing responses
            config: Optional configuration for the task
            evaluator_name: Optional name of the evaluator
            evaluator_config: Optional configuration for the evaluator
            **kwargs: Additional keyword arguments
        """
        self.prompt = prompt
        self.setup = setup or config or None
        self.extract = extract
        if evaluator:
            self.evaluator = evaluator
        else:
            self.evaluator = self._load_evaluator(evaluator_config)

    def _load_evaluator(self, evaluator_config: dict[str, Any] | None) -> Evaluator:
        if evaluator_config is None:
            evaluator_config = {}
        if "name" in evaluator_config:
            return EVALUATORS[evaluator_config["name"]](**evaluator_config)
        else:
            return Passthrough()
    
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
        return cls(
            prompt=data["prompt"],
            setup=data.get("setup", {}),
            extract=data.get("extract"),
            evaluator_config=data.get("evaluator_config"),
        )
        
    def to_dict(self) -> dict[str, Any]:
        """Convert Task to a dictionary.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "prompt": self.prompt,
            "setup": self.setup,
            "extract": self.extract,
            "evaluator_config": self.evaluator.to_dict(),
        }
    
    def __str__(self) -> str:
        return self.prompt
    
    def __repr__(self) -> str:
        return f"Task(prompt={self.prompt}, setup={self.setup}, extract={self.extract})"
