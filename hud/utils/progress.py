from __future__ import annotations

import time
from collections import defaultdict


class StepProgressTracker:
    """
    Tracks progress across potentially parallel async tasks based on steps completed.
    Provides estimates assuming tasks run up to max_steps_per_task.
    """
    def __init__(self, total_tasks: int, max_steps_per_task: int) -> None:
        if total_tasks <= 0:
            raise ValueError("total_tasks must be positive")
        if max_steps_per_task <= 0:
            raise ValueError("max_steps_per_task must be positive")
            
        self.total_tasks = total_tasks
        self.max_steps_per_task = max_steps_per_task
        self.total_potential_steps = total_tasks * max_steps_per_task

        # Use asyncio.Lock for potentially concurrent updates/reads if needed,
        # but start without for simplicity in single-threaded asyncio.
        # self._lock = asyncio.Lock()
        self._task_steps: dict[str, int] = defaultdict(int)
        self._finished_tasks: dict[str, bool] = defaultdict(bool)
        self._tasks_started = 0
        self._tasks_finished = 0
        
        self.start_time: float | None = None
        self.current_total_steps = 0

    def start_task(self, task_id: str) -> None:
        # async with self._lock: # If using lock
        if self.start_time is None:
            self.start_time = time.monotonic()
        self._task_steps[task_id] = 0
        self._finished_tasks[task_id] = False
        self._tasks_started += 1

    def increment_step(self, task_id: str) -> None:
        # async with self._lock:
        if (not self._finished_tasks[task_id] and
            self._task_steps[task_id] < self.max_steps_per_task):
            self._task_steps[task_id] += 1
            # Update overall progress immediately
            self._update_total_steps()

    def finish_task(self, task_id: str) -> None:
        # async with self._lock:
        if not self._finished_tasks[task_id]:
            # For calculation, consider a finished task as having completed max steps
            self._task_steps[task_id] = self.max_steps_per_task
            self._finished_tasks[task_id] = True
            self._tasks_finished += 1
            # Update overall progress
            self._update_total_steps()
            
    def _update_total_steps(self) -> None:
        # This could be expensive if called extremely frequently.
        # Called after increment or finish.
        # async with self._lock:
        self.current_total_steps = sum(self._task_steps.values())

    def get_progress(self) -> tuple[int, int, float]:
        """Returns (current_steps, total_potential_steps, percentage)."""
        # async with self._lock:
        # Recalculate here for safety, though _update_total_steps should keep it current
        # current_steps = sum(self._task_steps.values())
        current_steps = self.current_total_steps
        
        percentage = 0.0
        if self.total_potential_steps > 0:
            percentage = (current_steps / self.total_potential_steps) * 100
        return current_steps, self.total_potential_steps, percentage

    def get_stats(self) -> tuple[float, float | None]:
        """Returns (rate_steps_per_minute, eta_seconds_upper_bound)."""
        # async with self._lock:
        if self.start_time is None or self._tasks_started == 0:
            return 0.0, None # No rate or ETA yet

        elapsed_time = time.monotonic() - self.start_time
        current_steps = self.current_total_steps

        rate_sec = 0.0
        if elapsed_time > 0:
            rate_sec = current_steps / elapsed_time
        
        rate_min = rate_sec * 60 # Convert rate to steps per minute

        eta = None
        # ETA calculation still uses rate_sec (steps/second) for time estimation in seconds
        if rate_sec > 0:
            remaining_steps = self.total_potential_steps - current_steps
            eta = remaining_steps / rate_sec if remaining_steps > 0 else 0.0
        
        return rate_min, eta # Return rate in steps/min

    def is_finished(self) -> bool:
         # async with self._lock:
         return self._tasks_finished >= self.total_tasks

    def display(self, bar_length: int = 40) -> str:
        """Generates a progress string similar to tqdm."""
        current_steps, total_steps, percentage = self.get_progress()
        rate_min, eta = self.get_stats() # Rate is now per minute
        
        # Ensure valid values for display
        current_steps = min(current_steps, total_steps)
        percentage = max(0.0, min(100.0, percentage))

        filled_length = int(bar_length * current_steps // total_steps) if total_steps else 0
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        # Format time
        elapsed_str = "0:00"
        eta_str = "??:??"
        if self.start_time:
            elapsed_seconds = int(time.monotonic() - self.start_time)
            elapsed_str = f"{elapsed_seconds // 60}:{elapsed_seconds % 60:02d}"
            if eta is not None:
                 eta_seconds = int(eta)
                 eta_str = f"{eta_seconds // 60}:{eta_seconds % 60:02d}"
            elif self.is_finished():
                 eta_str = "0:00"
        
        # Update rate string format
        rate_str = f"{rate_min:.1f} steps/min" if rate_min > 0 else "?? steps/min"
        
        # Format steps - use K/M for large numbers if desired, keep simple for now
        steps_str = f"{current_steps}/{total_steps}"

        # tasks_str = f" {self._tasks_finished}/{self.total_tasks} tasks" # Optional tasks counter
        
        return f"{percentage:3.0f}%|{bar}| {steps_str} [{elapsed_str}<{eta_str}, {rate_str}]"
