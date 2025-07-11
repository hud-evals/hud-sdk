"""Statistics tracking and visualization for RL training."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TimingStats:
    """Track timing statistics with moving averages."""
    samples: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, duration: float) -> None:
        """Add a timing sample."""
        self.samples.append(duration)
    
    @property
    def mean(self) -> float:
        """Get mean duration."""
        return float(np.mean(self.samples)) if self.samples else 0.0
    
    @property
    def std(self) -> float:
        """Get standard deviation."""
        return float(np.std(self.samples)) if len(self.samples) > 1 else 0.0
    
    @property
    def count(self) -> int:
        """Get number of samples."""
        return len(self.samples)


@dataclass
class GRPOStats:
    """GRPO-specific statistics."""
    group_rewards: defaultdict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    group_completion_times: TimingStats = field(default_factory=TimingStats)
    buffer_sizes: deque = field(default_factory=lambda: deque(maxlen=100))
    advantages: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Track rewards within each group for variance calculation
    recent_group_variances: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Track task accuracies
    task_successes: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    task_attempts: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_group(self, task_id: str, rewards: List[float], completion_time: float) -> None:
        """Record statistics for a completed group."""
        self.group_rewards[task_id].extend(rewards)
        self.group_completion_times.add(completion_time)
        
        # Calculate and store variance for this specific group
        if len(rewards) >= 2:
            group_variance = float(np.var(rewards))
            self.recent_group_variances.append(group_variance)
        
        # Track task accuracy
        successes = sum(1 for r in rewards if r > 0)
        self.task_successes[task_id] = successes
        self.task_attempts[task_id] = len(rewards)
        
        # Calculate advantages for this group
        mean_reward = np.mean(rewards)
        for r in rewards:
            self.advantages.append(r - mean_reward)
    
    @property
    def avg_reward_variance(self) -> float:
        """Average within-group reward variance."""
        return float(np.mean(self.recent_group_variances)) if self.recent_group_variances else 0.0
    
    def get_task_accuracy_distribution(self) -> Dict[str, Any]:
        """Get distribution of task accuracies."""
        accuracies = []
        for task_id in self.task_attempts:
            if self.task_attempts[task_id] > 0:
                accuracy = self.task_successes[task_id] / self.task_attempts[task_id]
                accuracies.append(accuracy)
        
        if not accuracies:
            return {
                "buckets": [0, 0, 0, 0, 0, 0],
                "mean": 0.0,
                "count": 0
            }
        
        # Create buckets: 0%, 0-25%, 25-50%, 50-75%, 75-100%, 100%
        buckets = [0, 0, 0, 0, 0, 0]
        for acc in accuracies:
            if acc == 0:
                buckets[0] += 1
            elif acc < 0.25:
                buckets[1] += 1
            elif acc < 0.5:
                buckets[2] += 1
            elif acc < 0.75:
                buckets[3] += 1
            elif acc < 1.0:
                buckets[4] += 1
            else:
                buckets[5] += 1
        
        return {
            "buckets": buckets,
            "mean": np.mean(accuracies),
            "count": len(accuracies)
        }


class RLStatsTracker:
    """Comprehensive statistics tracking for RL training."""
    
    def __init__(self, max_concurrent: int, K: int, buffer_min: int):
        self.max_concurrent = max_concurrent
        self.K = K
        self.buffer_min = buffer_min
        
        # Start time
        self.start_time = time.time()
        
        # Worker statistics
        self.active_workers = 0
        self.worker_utilization: deque = deque(maxlen=100)
        
        # Queue statistics
        self.work_queue_sizes: deque = deque(maxlen=100)
        self.result_queue_sizes: deque = deque(maxlen=100)
        self.peak_work_queue = 0
        self.peak_result_queue = 0
        
        # Timing statistics
        self.episode_times = TimingStats()
        self.env_setup_times = TimingStats()
        self.agent_inference_times = TimingStats()
        self.env_step_times = TimingStats()
        self.evaluation_times = TimingStats()
        self.update_times = TimingStats()
        self.steps_per_episode = TimingStats()  # Track steps per episode
        self.network_times = TimingStats()  # Track network round-trip time
        self.model_inference_times = TimingStats()  # Track actual model inference time
        
        # GRPO-specific stats
        self.grpo_stats = GRPOStats()
        
        # Throughput tracking
        self.episodes_completed = 0
        self.updates_completed = 0
        self.episodes_per_second: deque = deque(maxlen=60)
        self.last_throughput_time = time.time()
        self.last_episode_count = 0
        
        # Container statistics
        self.container_creations = 0
        self.container_failures = 0
        
        # Training metrics
        self.training_losses: deque = deque(maxlen=100)
        self.kl_divergences: deque = deque(maxlen=100)
        self.entropies: deque = deque(maxlen=100)
        self.clip_fractions: deque = deque(maxlen=100)
        
        # Store full update statistics for detailed analysis
        self.update_stats: list[dict[str, float]] = []
        
        # Progress tracking
        self.total_tasks = 0
        self.total_epochs = 0.0
        self.tasks_completed = set()  # Track unique tasks completed
        
    def set_training_plan(self, num_tasks: int, num_epochs: float) -> None:
        """Set the total training plan for progress tracking."""
        self.total_tasks = num_tasks
        self.total_epochs = num_epochs
        
    def record_worker_active(self) -> None:
        """Record a worker becoming active."""
        self.active_workers += 1
        
    def record_worker_idle(self) -> None:
        """Record a worker becoming idle."""
        self.active_workers = max(0, self.active_workers - 1)
        
    def record_utilization(self) -> None:
        """Record current utilization."""
        utilization = self.active_workers / self.max_concurrent if self.max_concurrent > 0 else 0
        self.worker_utilization.append(utilization)
        
    def record_queue_sizes(self, work_queue_size: int, result_queue_size: int) -> None:
        """Record queue sizes."""
        self.work_queue_sizes.append(work_queue_size)
        self.result_queue_sizes.append(result_queue_size)
        self.peak_work_queue = max(self.peak_work_queue, work_queue_size)
        self.peak_result_queue = max(self.peak_result_queue, result_queue_size)
        
    def record_episode(
        self,
        total_time: float,
        setup_time: Optional[float] = None,
        inference_time: Optional[float] = None,
        step_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> None:
        """Record episode timing."""
        self.episode_times.add(total_time)
        if setup_time:
            self.env_setup_times.add(setup_time)
        if inference_time:
            self.agent_inference_times.add(inference_time)
        if step_time:
            self.env_step_times.add(step_time)
        if eval_time:
            self.evaluation_times.add(eval_time)
        if num_steps is not None:
            self.steps_per_episode.add(float(num_steps))
            
        self.episodes_completed += 1
        
        # Update throughput
        current_time = time.time()
        if current_time - self.last_throughput_time >= 1.0:
            eps = (self.episodes_completed - self.last_episode_count) / (
                current_time - self.last_throughput_time
            )
            self.episodes_per_second.append(eps)
            self.last_throughput_time = current_time
            self.last_episode_count = self.episodes_completed
            
    def record_update(self, duration: float, stats: Dict[str, float]) -> None:
        """Record gradient update."""
        self.update_times.add(duration)
        self.updates_completed += 1
        
        # Store full stats dict for later analysis
        self.update_stats.append(stats)
        
        # Record training metrics
        if "loss" in stats:
            self.training_losses.append(stats["loss"])
        if "kl" in stats:
            self.kl_divergences.append(stats["kl"])
        if "entropy" in stats:
            self.entropies.append(stats["entropy"])
        if "clip_frac" in stats:
            self.clip_fractions.append(stats["clip_frac"])
            
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics summary."""
        elapsed = time.time() - self.start_time
        
        # Calculate parallelization efficiency
        avg_utilization = np.mean(self.worker_utilization) if self.worker_utilization else 0
        ideal_episodes = self.max_concurrent * elapsed / self.episode_times.mean if self.episode_times.mean > 0 else 0
        parallel_efficiency = self.episodes_completed / ideal_episodes if ideal_episodes > 0 else 0
        
        # Get task accuracy distribution
        task_acc_dist = self.grpo_stats.get_task_accuracy_distribution()
        
        # Calculate progress
        epoch_progress = 0.0
        if self.total_tasks > 0 and self.K > 0:
            samples_per_epoch = self.total_tasks * self.K
            epoch_progress = self.episodes_completed / samples_per_epoch
        
        return {
            "elapsed_time": elapsed,
            "progress": {
                "episodes_completed": self.episodes_completed,
                "unique_tasks_seen": len(self.tasks_completed),
                "total_tasks": self.total_tasks,
                "epoch_progress": epoch_progress,
                "target_epochs": self.total_epochs,
            },
            "parallelization": {
                "active_workers": self.active_workers,
                "max_concurrent": self.max_concurrent,
                "avg_utilization": avg_utilization,
                "parallel_efficiency": parallel_efficiency,
                "avg_work_queue": np.mean(self.work_queue_sizes) if self.work_queue_sizes else 0,
                "avg_result_queue": np.mean(self.result_queue_sizes) if self.result_queue_sizes else 0,
            },
            "performance": {
                "episodes_completed": self.episodes_completed,
                "updates_completed": self.updates_completed,
                "episodes_per_second": np.mean(self.episodes_per_second) if self.episodes_per_second else 0,
                "updates_per_minute": self.updates_completed / elapsed * 60 if elapsed > 0 else 0,
            },
            "timing": {
                "episode_time": self.episode_times.mean,
                "env_setup_time": self.env_setup_times.mean,
                "agent_inference_time": self.agent_inference_times.mean,
                "env_step_time": self.env_step_times.mean,
                "evaluation_time": self.evaluation_times.mean,
                "update_time": self.update_times.mean,
                "steps_per_episode": self.steps_per_episode.mean,
                "network_time": self.network_times.mean,
                "model_inference_time": self.model_inference_times.mean,
            },
            "grpo": {
                "avg_group_completion": self.grpo_stats.group_completion_times.mean,
                "avg_reward_variance": self.grpo_stats.avg_reward_variance,
                "advantage_std": np.std(self.grpo_stats.advantages) if self.grpo_stats.advantages else 0,
                "buffer_fill_rate": np.mean(self.grpo_stats.buffer_sizes) / self.buffer_min if self.grpo_stats.buffer_sizes else 0,
            },
            "task_accuracy": task_acc_dist,
            "training": {
                "avg_loss": np.mean(self.training_losses) if self.training_losses else 0,
                "avg_pg_loss": np.mean([s.get("pg_loss", 0) for s in self.update_stats]) if hasattr(self, "update_stats") and self.update_stats else 0,
                "avg_kl": np.mean(self.kl_divergences) if self.kl_divergences else 0,
                "avg_approx_kl": np.mean([s.get("approx_kl", 0) for s in self.update_stats]) if hasattr(self, "update_stats") and self.update_stats else 0,
                "avg_entropy": np.mean(self.entropies) if self.entropies else 0,
                "avg_clip_frac": np.mean(self.clip_fractions) if self.clip_fractions else 0,
                "avg_ratio": np.mean([s.get("avg_ratio", 1.0) for s in self.update_stats]) if hasattr(self, "update_stats") and self.update_stats else 1.0,
                "avg_grad_abs": np.mean([s.get("avg_grad_abs", 0) for s in self.update_stats]) if hasattr(self, "update_stats") else 0,
            },
            "containers": {
                "created": self.container_creations,
                "failed": self.container_failures,
            }
        }
        
    def format_dashboard(self) -> str:
        """Format statistics as a nice terminal dashboard."""
        stats = self.get_summary()
        elapsed = stats["elapsed_time"]
        
        # Format elapsed time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Shorter format for better readability
        p = stats['parallelization']
        perf = stats['performance']
        t = stats['timing']
        g = stats['grpo']
        tr = stats['training']
        ta = stats['task_accuracy']
        prog = stats['progress']
        
        # Create progress bar
        progress_pct = (prog['epoch_progress'] / prog['target_epochs'] * 100) if prog['target_epochs'] > 0 else 0
        progress_filled = int(progress_pct / 100 * 30)
        progress_bar = "█" * progress_filled + "─" * (30 - progress_filled)
        
        # Format task accuracy buckets as a histogram
        buckets = ta['buckets']
        total_tasks = ta['count']
        
        # Create histogram bars
        def make_bar(count, total, width=10):
            if total == 0:
                return "─" * width
            pct = count / total
            filled = int(pct * width)
            return "█" * filled + "─" * (width - filled)
        
        # Calculate per-step averages (episodes typically have multiple steps)
        # Use actual tracked steps per episode if available
        steps_per_episode = t.get('steps_per_episode', 1)
        if steps_per_episode <= 0:
            steps_per_episode = 1
        
        # Calculate per-step times
        # Note: agent_inference_time and env_step_time are already totals for the episode
        # They were accumulated in default_run_episode, so they represent the sum across all steps
        infer_per_step = t['agent_inference_time'] / steps_per_episode
        step_per_step = t['env_step_time'] / steps_per_episode
        
        # Network and model times are already per-step averages (not totals)
        network_per_step = t.get('network_time', 0)
        model_per_step = t.get('model_inference_time', 0)
        
        dashboard = f"""
┌─ GRPO Training [{time_str}] ───────────────────────────────────────────────────────────────────────────────┐
│ Progress: [{progress_bar}] {progress_pct:5.1f}% │ Epoch: {prog['epoch_progress']:.2f}/{prog['target_epochs']:.1f} │ Tasks: {prog['unique_tasks_seen']}/{prog['total_tasks']} │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Workers: {p['active_workers']}/{self.max_concurrent} ({p['avg_utilization']*100:3.0f}%) │ Episodes: {perf['episodes_completed']:4d} ({perf['episodes_per_second']:4.1f}/s) │ Updates: {perf['updates_completed']:3d} ({perf['updates_per_minute']:4.1f}/m) │ Efficiency: {p['parallel_efficiency']*100:3.0f}% │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Episode Total: {t['episode_time']*1000:5.0f}ms │ Setup: {t['env_setup_time']*1000:4.0f}ms │ Eval: {t['evaluation_time']*1000:4.0f}ms │ Avg Steps: {steps_per_episode:.1f} │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Per Step: Total: {infer_per_step*1000:4.0f}ms │ Network: {network_per_step*1000:4.0f}ms │ Model: {model_per_step*1000:4.0f}ms │ Env: {step_per_step*1000:4.0f}ms │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Group[K={self.K}]: {g['avg_group_completion']*1000:5.0f}ms │ RewVar: {g['avg_reward_variance']:5.3f} │ AdvStd: {g['advantage_std']:5.3f} │ Buffer: {g['buffer_fill_rate']*100:3.0f}% │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Loss: {tr['avg_loss']:6.3f} │ KL: {tr['avg_kl']:6.3f} │ Entropy: {tr['avg_entropy']:6.3f} │ Clip: {tr['avg_clip_frac']:5.3f} │ Update: {t['update_time']*1000:5.0f}ms │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Task Accuracy Distribution (n={total_tasks}) │ Mean: {ta['mean']*100:5.1f}%                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│     0%: {make_bar(buckets[0], total_tasks)} ({buckets[0]:3d}) │ 0-25%: {make_bar(buckets[1], total_tasks)} ({buckets[1]:3d}) │ 25-50%: {make_bar(buckets[2], total_tasks)} ({buckets[2]:3d})                            │
│ 50-75%: {make_bar(buckets[3], total_tasks)} ({buckets[3]:3d}) │ 75-99%: {make_bar(buckets[4], total_tasks)} ({buckets[4]:3d}) │ 100%: {make_bar(buckets[5], total_tasks)} ({buckets[5]:3d})                             │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        
        return dashboard.strip()
