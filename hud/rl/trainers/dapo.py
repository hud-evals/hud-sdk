"""Decoupled Advantage Policy Optimization (DAPO) trainer."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

from ..types import Batch, Trajectory
from .base import TrainerBase

logger = logging.getLogger(__name__)


@dataclass
class _DAPOSample:
    """Internal sample format for DAPO buffer."""
    trajectory: Trajectory
    reward: float
    advantage: float
    weight: float  # dynamic sampling weight


class DAPOTrainer(TrainerBase):
    """DAPO trainer with dynamic sampling and asymmetric clipping.
    
    Based on "Diversity-Aware Policy Optimization" (DAPO) paper:
    - Groups K trajectories for the same prompt
    - Computes dynamic weights based on advantages
    - Uses Clip-Higher for asymmetric clipping to promote diversity
    - Applies overlong penalty to reduce reward noise
    - Token-level policy gradient (implemented in VLMAgent.update)
    """
    
    def __init__(
        self, 
        *args, 
        lambda_temp: float = 2.0,
        clip_epsilon_high: float = 0.4,
        clip_epsilon_low: float = 0.2,
        overlong_penalty_factor: float = 0.1,
        overlong_threshold: int = 200,
        **kwargs
    ):
        """Initialize DAPO trainer.
        
        Args:
            lambda_temp: Temperature for dynamic weight calculation (default: 2.0 from paper)
            clip_epsilon_high: Upper bound for clipping (default: 0.4 for 1+2ε)
            clip_epsilon_low: Lower bound for clipping (default: 0.2 for 1-ε)
            overlong_penalty_factor: Penalty factor for overlong responses
            overlong_threshold: Token count threshold for overlong penalty
            *args, **kwargs: Passed to TrainerBase
        """
        super().__init__(*args, **kwargs)
        self.lambda_temp = lambda_temp
        self.clip_epsilon_high = clip_epsilon_high
        self.clip_epsilon_low = clip_epsilon_low
        self.overlong_penalty_factor = overlong_penalty_factor
        self.overlong_threshold = overlong_threshold
    
    def _process_group(self, group: List[Tuple[Trajectory, float]]) -> List[_DAPOSample]:
        """Compute advantages and dynamic weights for a group.
        
        Args:
            group: K trajectories for the same task
            
        Returns:
            List of samples with advantages and weights
        """
        # Extract rewards and apply overlong penalty
        adjusted_rewards = []
        for traj, reward in group:
            # Count total tokens in response
            total_tokens = sum(
                len(t.action_sample.tokens) if t.action_sample and t.action_sample.tokens else 0
                for t in traj.transitions
            )
            
            # Apply overlong penalty if response is too long
            if total_tokens > self.overlong_threshold:
                penalty = self.overlong_penalty_factor * (total_tokens - self.overlong_threshold) / self.overlong_threshold
                adjusted_reward = reward * (1 - penalty)
            else:
                adjusted_reward = reward
            
            adjusted_rewards.append(adjusted_reward)
        
        mean_reward = sum(adjusted_rewards) / len(adjusted_rewards)
        
        # Compute advantages
        advantages = [r - mean_reward for r in adjusted_rewards]
        
        # Compute dynamic weights: w_i = exp(λ * advantage_i)
        weights = [math.exp(self.lambda_temp * adv) for adv in advantages]
        
        # Normalize weights to sum to K (maintains effective batch size)
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w * self.K / weight_sum for w in weights]
        else:
            # Fallback to uniform weights if sum is 0
            weights = [1.0] * len(weights)
        
        # Create samples
        samples = []
        for i, ((traj, reward), advantage, weight) in enumerate(zip(group, advantages, weights)):
            samples.append(_DAPOSample(
                trajectory=traj,
                reward=adjusted_rewards[i],  # Use adjusted reward
                advantage=advantage,
                weight=weight
            ))
            
        return samples
    
    def _create_batch(self, samples: List[_DAPOSample]) -> Batch:
        """Create text-based training batch with DAPO weights."""
        observations = []
        texts = []
        advantages = []
        returns = []
        weights = []
        old_log_probs = []
        actions = []  # Keep for debugging/analysis
        
        for sample in samples:
            for transition in sample.trajectory.transitions:
                observations.append(transition.observation)
                texts.append(transition.text)
                advantages.append(sample.advantage)
                returns.append(transition.reward)
                # Apply trajectory weight to each step
                weights.append(sample.weight)
                
                # Include actions for analysis if available
                if transition.actions:
                    actions.append(transition.actions)
                
                # Store log prob if available
                if transition.log_prob is not None:
                    old_log_probs.append(transition.log_prob)
        
        # Validate we have text data
        if not texts or not all(texts):
            raise ValueError("DAPO requires text data from agent.sample()")
                
        return Batch(
            observations=observations,
            texts=texts,
            advantages=advantages,
            returns=returns,
            old_log_probs=old_log_probs if old_log_probs else None,
            actions=actions if actions else None,
            metadata={
                "algorithm": "DAPO",
                "num_trajectories": len(samples),
                "num_transitions": len(observations),
                "weights": weights,  # Pass to optimizer
                "lambda_temp": self.lambda_temp,
                "clip_epsilon_high": self.clip_epsilon_high,
                "clip_epsilon_low": self.clip_epsilon_low,
            }
        ) 