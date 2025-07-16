"""Base trainer class for RL algorithms."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Iterable, List, Tuple

from hud.agent import Agent
from hud.gym import make as gym_make
from hud.task import Task

from ..stats import RLStatsTracker
from ..types import Batch, Trajectory, RLTransition, TaskQueue

logger = logging.getLogger(__name__)


async def default_run_episode(
    agent: Agent,
    task: Task,
    *,
    max_steps: int = 64,
    stats_tracker: RLStatsTracker | None = None,
) -> Tuple[Trajectory, float]:
    """Default episode runner that collects ActionSamples for RL training.
    
    Uses agent.sample() to get full information needed for RL training,
    including generated text and log probabilities.
    
    Args:
        agent: Agent to run (must implement sample())
        task: Task containing gym spec and metadata
        max_steps: Maximum steps before forced termination
        stats_tracker: Optional stats tracker for timing
        
    Returns:
        (trajectory, reward) tuple with full ActionSample data
    """
    episode_start = time.time()
    setup_start = time.time()
    
    env = await gym_make(task)
    setup_time = time.time() - setup_start
    
    try:
        obs, _ = await env.reset()
        transitions = []
        
        total_inference_time = 0
        total_step_time = 0
        
        for step in range(max_steps):
            # Time agent inference - use sample() for full RL data
            inference_start = time.time()
            action_sample = await agent.sample(obs)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Extract detailed timing from metadata if available
            if stats_tracker and action_sample.metadata and 'timing' in action_sample.metadata:
                timing = action_sample.metadata['timing']
                if 'network_ms' in timing:
                    stats_tracker.network_times.add(timing['network_ms'] / 1000.0)  # Convert to seconds
                if 'model_generate_ms' in timing:
                    stats_tracker.model_inference_times.add(timing['model_generate_ms'] / 1000.0)
            
            # Extract actions for environment
            actions = action_sample.actions or []
            
            # Time environment step
            step_start = time.time()
            next_obs, _, env_done, _ = await env.step(actions)
            step_time = time.time() - step_start
            total_step_time += step_time
            
            # Create RL transition with full sample
            transitions.append(RLTransition(
                observation=obs,
                action_sample=action_sample,
                reward=0.0,
                next_observation=next_obs  # Store next observation too
            ))
            
            obs = next_obs
            if action_sample.done or env_done:
                break
                
        # Time evaluation
        eval_start = time.time()
        result = await env.evaluate()
        eval_time = time.time() - eval_start
        
        reward = float(result.get("reward", 0.0))
        
        # Mark final transition
        if transitions:
            transitions[-1].reward = reward
            transitions[-1].done = True
            
        # Record episode statistics
        if stats_tracker:
            total_time = time.time() - episode_start
            num_steps = len(transitions)  # Track number of steps taken
            stats_tracker.record_episode(
                total_time=total_time,
                setup_time=setup_time,
                inference_time=total_inference_time,
                step_time=total_step_time,
                eval_time=eval_time,
                num_steps=num_steps  # Pass number of steps
            )
            
        return Trajectory(
            transitions=transitions,
            total_reward=reward,
            task_id=task.id,
        ), reward
        
    finally:
        await env.close()


class TrainerBase(ABC):
    """Base class for RL trainers with semaphore-based concurrency.
    
    Requires agents that implement the sample() method for full RL support.
    """
    
    def __init__(
        self,
        agent: Agent,
        dataset: Iterable[Task],
        *,
        K: int = 8,
        max_concurrent: int = 64,
        buffer_min: int = 256,
        batch_size: int = 32,
        run_episode: Callable | None = None,
        show_dashboard: bool = True,
        dashboard_interval: float = 2.0,
    ):
        # Validate agent supports sample() and update()
        if not hasattr(agent, 'sample') or not callable(getattr(agent, 'sample')):
            raise ValueError(
                f"Agent {type(agent).__name__} does not implement sample() method. "
                "RL training requires agents that can return ActionSample with text and log probabilities."
            )
        
        if not hasattr(agent, 'update') or not callable(getattr(agent, 'update')):
            raise ValueError(
                f"Agent {type(agent).__name__} does not implement update() method. "
                "RL training requires agents that can perform gradient updates on batches."
            )
            
        self.agent = agent
        self.task_queue = TaskQueue(dataset)
        self.K = K
        self.max_concurrent = max_concurrent
        self.buffer_min = buffer_min
        self.batch_size = batch_size
        self.run_episode = run_episode or default_run_episode
        self.show_dashboard = show_dashboard
        self.dashboard_interval = dashboard_interval
        
        # Queues
        self.work_queue: asyncio.Queue[Task] = asyncio.Queue()
        self.result_queue: asyncio.Queue[Tuple[str | None, Trajectory, float]] = asyncio.Queue()
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # State
        self._running = False
        self._stats = defaultdict(float)
        
        # Initialize statistics tracker
        self.stats_tracker = RLStatsTracker(max_concurrent, K, buffer_min)
        
        # Calculate dataset size
        if isinstance(dataset, list):
            self._dataset_size = len(dataset)
        else:
            # Try to get length if possible
            try:
                self._dataset_size = len(dataset)  # type: ignore
            except TypeError:
                self._dataset_size = None  # Unknown size for iterators
        
    def calculate_updates_per_epoch(self) -> int:
        """Calculate number of updates per epoch based on dataset and batch configuration.
        
        An epoch is one complete pass through all tasks with K samples each.
        """
        if self._dataset_size is None:
            raise ValueError("Cannot calculate updates for infinite dataset")
            
        # Total samples per epoch = num_tasks * K
        samples_per_epoch = self._dataset_size * self.K
        
        # Updates per epoch = floor(samples / batch_size)
        updates_per_epoch = samples_per_epoch // self.batch_size
        
        return updates_per_epoch
        
    @abstractmethod
    def _process_group(self, group: List[Tuple[Trajectory, float]]) -> List[Any]:
        """Process a group of K trajectories into training samples.
        
        Args:
            group: List of (trajectory, reward) tuples for the same task
            
        Returns:
            List of processed samples to add to buffer
        """
        pass
        
    async def train(self, num_epochs: float = 1.0) -> dict[str, float]:
        """Run training for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs (can be fractional)
            
        Returns:
            Dict of training statistics
        """
        self._running = True
        self._stats.clear()
        
        updates_per_epoch = self.calculate_updates_per_epoch()
        total_updates = int(num_epochs * updates_per_epoch)
        
        # Total samples we actually want this run
        total_samples_needed = int(self._dataset_size * self.K * num_epochs) if self._dataset_size else None

        # Set training plan in stats tracker
        if self._dataset_size is not None:
            self.stats_tracker.set_training_plan(self._dataset_size, num_epochs)

        # Store target samples so producer/workers can respect it
        self._target_samples = total_samples_needed
        
        if not self.show_dashboard and self._dataset_size is not None:
            logger.info(
                "Training for %.1f epochs (%d tasks × K=%d × %.1f epochs = %d samples, %d updates)",
                num_epochs, self._dataset_size, self.K, num_epochs,
                int(self._dataset_size * self.K * num_epochs), total_updates
            )
        
        # Suppress logs if dashboard is enabled
        original_log_levels = {}
        if self.show_dashboard:
            # Save original log levels and set to WARNING or higher
            for logger_name in ['hud.gym', 'hud.env', 'hud.rl', 'asyncio']:
                logger_obj = logging.getLogger(logger_name)
                original_log_levels[logger_name] = logger_obj.level
                logger_obj.setLevel(logging.WARNING)
        
        # Start producer that feeds K copies of each task
        producer = asyncio.create_task(self._produce_tasks())
        
        # Start workers
        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_concurrent)
        ]
        
        # Start dashboard updater if enabled
        dashboard_task = None
        if self.show_dashboard:
            dashboard_task = asyncio.create_task(self._update_dashboard())
        
        # Start stats updater
        stats_task = asyncio.create_task(self._update_stats())
        
        # Run consumer until done
        try:
            await self._consume_results(total_updates)
        finally:
            # Cleanup
            self._running = False
            producer.cancel()
            for w in workers:
                w.cancel()
            if dashboard_task:
                dashboard_task.cancel()
            stats_task.cancel()
            
            # Wait for cancellation
            tasks_to_wait = [producer, *workers, stats_task]
            if dashboard_task:
                tasks_to_wait.append(dashboard_task)
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)
            
            # Restore original log levels
            if self.show_dashboard:
                for logger_name, level in original_log_levels.items():
                    logging.getLogger(logger_name).setLevel(level)
            
            # Show final dashboard
            if self.show_dashboard:
                print("\n" + self.stats_tracker.format_dashboard())
            
        return dict(self._stats)
        
    async def _produce_tasks(self) -> None:
        """Feed K copies of each task to work queue."""
        try:
            task_counter = 0
            produced = 0
            async for task in self.task_queue:
                if self._target_samples is not None and produced >= self._target_samples:
                    break
                # Ensure task has an ID
                if not hasattr(task, 'id') or task.id is None:
                    task.id = f"task_{task_counter}"
                    task_counter += 1
                # How many copies of this task should we enqueue?
                copies = self.K
                if self._target_samples is not None and produced + copies > self._target_samples:
                    copies = self._target_samples - produced
                for _ in range(copies):
                    await self.work_queue.put(task)
                    produced += 1
                if self._target_samples is not None and produced >= self._target_samples:
                    break
        except asyncio.CancelledError:
            pass
            
    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that runs episodes."""
        while self._running:
            try:
                # Get task with timeout to check _running periodically
                task = await asyncio.wait_for(self.work_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
                
            async with self.semaphore:
                self.stats_tracker.record_worker_active()
                try:
                    # Pass stats tracker to run_episode if it's our default
                    if self.run_episode == default_run_episode:
                        traj, reward = await self.run_episode(
                            self.agent, task, stats_tracker=self.stats_tracker
                        )
                    else:
                        traj, reward = await self.run_episode(self.agent, task)
                        
                    await self.result_queue.put((task.id, traj, reward))
                    # Ensure global stats tracker counts this episode even when using a custom
                    # run_episode function that may not call record_episode().
                    self.stats_tracker.episodes_completed += 1
                except Exception:
                    if not self.show_dashboard:
                        logger.exception("Worker %d failed on task %s", worker_id, task.id)
                    self.stats_tracker.container_failures += 1
                finally:
                    self.stats_tracker.record_worker_idle()
                    
    async def _consume_results(self, total_updates: int) -> None:
        """Collect results and trigger updates when buffer is full."""
        groups = defaultdict(list)
        group_start_times = {}
        buffer = []
        updates_done = 0
        
        # Calculate expected total episodes
        expected_episodes = self._target_samples
        episodes_received = 0
        
        while updates_done < total_updates:
            # Check if we've hit our target episodes first (more important than updates)
            if expected_episodes is not None and episodes_received >= expected_episodes:
                if not self.show_dashboard:
                    logger.info(
                        "Reached target episodes (%d/%d). Processing final batch if needed.",
                        episodes_received, expected_episodes
                    )
                # Process any remaining buffer
                if buffer and len(buffer) >= self.batch_size:
                    if not self.show_dashboard:
                        logger.info(
                            "Processing final batch (%d samples in buffer)",
                            len(buffer)
                        )
                    batch = self._create_batch(buffer[:self.batch_size])
                    buffer = buffer[self.batch_size:]
                    
                    # Time the update
                    update_start = time.time()
                    stats = await self.agent.update(batch)
                    update_time = time.time() - update_start
                    
                    updates_done += 1
                    
                    # Record update statistics
                    self.stats_tracker.record_update(update_time, stats)
                    
                    # Log stats
                    for k, v in stats.items():
                        self._stats[k] += v
                
                # Log remaining buffer if not processed
                if buffer:
                    if not self.show_dashboard:
                        logger.info(
                            "Training complete. %d samples remaining in buffer (need %d for update)",
                            len(buffer), self.batch_size
                        )
                break
            
            try:
                # Use a timeout to avoid hanging forever
                task_id, traj, reward = await asyncio.wait_for(
                    self.result_queue.get(), 
                    timeout=5.0
                )
                episodes_received += 1
            except asyncio.TimeoutError:
                # Check if workers are still active
                if self.stats_tracker.active_workers == 0:
                    if not self.show_dashboard:
                        logger.info("No active workers and timeout reached, ending training")
                    break
                continue
            except asyncio.CancelledError:
                break
                
            # Skip if task_id is None
            if task_id is None:
                logger.warning("Received trajectory with no task_id, skipping")
                continue
                
            # Track task completion for progress
            self.stats_tracker.tasks_completed.add(task_id)
                
            # Track group formation time
            if task_id not in group_start_times:
                group_start_times[task_id] = time.time()
                
            groups[task_id].append((traj, reward))
            
            # Check if group is complete
            if len(groups[task_id]) == self.K:
                # Record group completion time
                completion_time = time.time() - group_start_times[task_id]
                rewards = [r for _, r in groups[task_id]]
                self.stats_tracker.grpo_stats.add_group(task_id, rewards, completion_time)
                
                # Process group and add to buffer
                samples = self._process_group(groups[task_id])
                buffer.extend(samples)
                del groups[task_id]
                del group_start_times[task_id]
                
                # Record buffer size
                self.stats_tracker.grpo_stats.buffer_sizes.append(len(buffer))
                
                if not self.show_dashboard:
                    logger.info(
                        "Completed group for task %s (buffer size: %d)",
                        task_id, len(buffer)
                    )
                
                # Update if buffer is full
                if len(buffer) >= self.buffer_min:
                    batch = self._create_batch(buffer[:self.batch_size])
                    buffer = buffer[self.batch_size:]
                    
                    # Time the update
                    update_start = time.time()
                    stats = await self.agent.update(batch)
                    update_time = time.time() - update_start
                    
                    updates_done += 1
                    
                    # Record update statistics
                    self.stats_tracker.record_update(update_time, stats)
                    
                    # Log stats with debugging
                    logger.info("=== DEBUG: Base Trainer Stats Accumulation ===")
                    for k, v in stats.items():
                        logger.info(f"  Accumulating {k}: {v} (type: {type(v)})")
                        try:
                            if k in self._stats:
                                logger.info(f"    Current _stats[{k}]: {self._stats[k]} (type: {type(self._stats[k])})")
                                self._stats[k] += v
                                logger.info(f"    New _stats[{k}]: {self._stats[k]} (type: {type(self._stats[k])})")
                            else:
                                logger.info(f"    Initializing _stats[{k}] = {v}")
                                self._stats[k] = v
                        except Exception as e:
                            logger.error(f"  ERROR accumulating {k}: {e}")
                            logger.error(f"    _stats[{k}]: {self._stats.get(k, 'KEY_NOT_FOUND')} (type: {type(self._stats.get(k, None))})")
                            logger.error(f"    stats[{k}]: {v} (type: {type(v)})")
                            raise
                    logger.info("=== END DEBUG ===")
                    
                    if not self.show_dashboard:
                        logger.info(
                            "Update %d/%d - %s",
                            updates_done, total_updates,
                            {k: f"{v:.4f}" for k, v in stats.items()}
                        )
                        
        # Process any remaining incomplete groups (log warning)
        if groups and not self.show_dashboard:
            logger.warning(
                "Training ended with %d incomplete groups (need K=%d samples each)",
                len(groups), self.K
            )
                    
    async def _update_stats(self) -> None:
        """Periodically update queue and utilization statistics."""
        while self._running:
            try:
                self.stats_tracker.record_utilization()
                self.stats_tracker.record_queue_sizes(
                    self.work_queue.qsize(),
                    self.result_queue.qsize()
                )
                await asyncio.sleep(0.1)  # Update every 100ms
            except asyncio.CancelledError:
                break
                
    async def _update_dashboard(self) -> None:
        """Periodically update the dashboard display."""
        while self._running:
            try:
                # Clear screen and show dashboard
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                print(self.stats_tracker.format_dashboard())
                await asyncio.sleep(self.dashboard_interval)
            except asyncio.CancelledError:
                break
                    
    @abstractmethod
    def _create_batch(self, samples: List[Any]) -> Batch:
        """Create training batch from samples."""
        pass 