"""Core data structures for RL training."""

from __future__ import annotations

from typing import Any, List, Optional, AsyncIterator, Iterable
import asyncio
from pydantic import BaseModel, Field, computed_field

from hud.task import Task
from hud.adapters import CLA
from hud.utils.common import Observation


class ActionSample(BaseModel):
    """Complete sample from agent including text, actions, and probabilities.
    
    This preserves all information needed for RL training while maintaining
    compatibility with the existing action-based interface.
    """
    # Raw LLM output
    text: str                                    # What the model actually generated
    log_probs: Optional[List[float]] = None      # Token-level log probabilities  
    tokens: Optional[List[str]] = None           # Tokens corresponding to log_probs
    total_log_prob: Optional[float] = None       # Sum of log_probs for sequence
    
    # Parsed and processed actions
    actions: Optional[List[CLA]] = None          # Executable actions after adapter processing
    raw_actions: Optional[List[dict | str]] = None     # Actions before adapter processing
    
    # Metadata
    done: bool = False                           # Whether agent thinks task is complete
    metadata: Optional[dict[str, Any]] = None    # Model-specific data (reasoning, etc)
    
    def model_post_init(self, __context) -> None:
        """Compute total log prob if not provided."""
        if self.total_log_prob is None and self.log_probs:
            self.total_log_prob = sum(self.log_probs)


class RLTransition(BaseModel):
    """Single step in an RL episode.
    
    Combines observation with the agent's full response and environment feedback.
    """
    observation: Observation                          # Input to the agent
    action_sample: ActionSample                       # Full agent response with text and probabilities
    reward: float = 0.0                              # Reward from environment (usually sparse)
    next_observation: Optional[Observation] = None    # For algorithms that need s'
    done: bool = False                               # Whether this transition ended the episode
    
    @computed_field
    @property
    def text(self) -> str:
        """Get generated text."""
        return self.action_sample.text
    
    @computed_field
    @property
    def actions(self) -> Optional[List[CLA]]:
        """Get executed actions."""
        return self.action_sample.actions
    
    @computed_field
    @property
    def log_prob(self) -> Optional[float]:
        """Get total log probability."""
        return self.action_sample.total_log_prob


class Trajectory(BaseModel):
    """Complete episode trajectory for RL training."""
    transitions: List[RLTransition]
    total_reward: float = 0.0
    task_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.transitions)
    
    @computed_field
    @property
    def has_text_data(self) -> bool:
        """Check if trajectory has text and log probs for language model training."""
        return all(
            t.action_sample.text and t.action_sample.total_log_prob is not None
            for t in self.transitions
        )


class Batch(BaseModel):
    """Training batch for policy optimization.
    
    Supports text-based RL training with language models.
    """
    observations: List[Observation]                   # Input observations
    texts: List[str]                                 # Generated text for each step
    advantages: List[float]                          # Advantage values
    returns: List[float]                             # Returns/rewards
    old_log_probs: Optional[List[float]] = None      # Log P(text|obs) under old policy
    
    # Optional fields for compatibility or extended algorithms
    actions: Optional[List[List[CLA]]] = None        # Parsed actions if needed
    
    metadata: Optional[dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.observations)
    
    @computed_field
    @property
    def is_valid_for_grpo(self) -> bool:
        """Check if batch has required data for GRPO training."""
        return (
            len(self.texts) == len(self.observations) and
            len(self.advantages) == len(self.observations) and
            len(self.returns) == len(self.observations)
        ) 


class TaskQueue:
    """Async iterator that yields tasks from a dataset."""
    
    def __init__(self, dataset: Iterable[Task]):
        self._dataset = dataset
        self._iter = iter(dataset)
        self._exhausted = False
    
    def __aiter__(self) -> AsyncIterator[Task]:
        return self
    
    async def __anext__(self) -> Task:
        if self._exhausted:
            raise StopAsyncIteration
        
        # Helper function to safely get next item
        def get_next():
            try:
                return next(self._iter)
            except StopIteration:
                return None
        
        # Run in executor to avoid blocking if dataset I/O is slow
        loop = asyncio.get_event_loop()
        task = await loop.run_in_executor(None, get_next)
        
        if task is None:
            self._exhausted = True
            raise StopAsyncIteration
            
        return task 
