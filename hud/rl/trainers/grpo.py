"""Grouped Relative Policy Optimization (GRPO) trainer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

from ..types import Batch, Trajectory
from .base import TrainerBase

logger = logging.getLogger(__name__)


@dataclass
class _GRPOSample:
    """Internal sample format for GRPO buffer."""
    trajectory: Trajectory
    reward: float
    advantage: float


class GRPOTrainer(TrainerBase):
    """GRPO trainer implementing grouped relative advantages.
    
    Based on "Group Relative Policy Optimization" (2024):
    - Groups K trajectories for the same prompt
    - Computes advantage as A_i = R_i - mean(R)
    - No advantage normalization for small K
    """
    
    def _process_group(self, group: List[Tuple[Trajectory, float]]) -> List[_GRPOSample]:
        """Compute relative advantages for a group.
        
        Args:
            group: K trajectories for the same task
            
        Returns:
            List of samples with computed advantages
        """
        # Extract rewards
        rewards = [r for _, r in group]
        mean_reward = sum(rewards) / len(rewards)
        
        # Compute advantages (no normalization per paper)
        samples = []
        for traj, reward in group:
            advantage = reward - mean_reward
            samples.append(_GRPOSample(
                trajectory=traj,
                reward=reward,
                advantage=advantage
            ))
        
        return samples
    
    def _create_batch(self, samples: List[_GRPOSample]) -> Batch:
        """Create text-based training batch for GRPO.
        
        Since we're using RLTransition with ActionSamples, we always have
        text and can create proper language model training batches.
        """
        observations = []
        texts = []
        advantages = []
        returns = []
        old_log_probs = []
        actions = []  # Keep for debugging/analysis
        
        for sample in samples:
            for transition in sample.trajectory.transitions:
                observations.append(transition.observation)
                texts.append(transition.text)
                advantages.append(sample.advantage)
                returns.append(transition.reward)
                
                # Include actions for analysis if available
                if transition.actions:
                    actions.append(transition.actions)
                
                # Store log prob if available
                if transition.log_prob is not None:
                    old_log_probs.append(transition.log_prob)
                
        # Validate we have text data
        if not texts or not all(texts):
            raise ValueError("GRPO requires text data from agent.sample()")
                
        return Batch(
            observations=observations,
            texts=texts,
            advantages=advantages,
            returns=returns,
            old_log_probs=old_log_probs if old_log_probs else None,
            actions=actions if actions else None,
            metadata={
                "algorithm": "GRPO", 
                "num_trajectories": len(samples),
                "num_transitions": len(observations)
            }
        ) 