#!/usr/bin/env python3
"""Performance test for GRPO training with system monitoring and metric tracking."""

import asyncio
import json
import logging
import os
import psutil
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np

from hud.agent.hosted_vlm_agent import HostedVLMAgent
from hud.rl import GRPOTrainer
from hud.task import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources during training."""
    
    def __init__(self):
        self.cpu_percent_history = []
        self.memory_percent_history = []
        self.memory_gb_history = []
        self.timestamps = []
        self.running = False
        
    async def start_monitoring(self, interval: float = 2.0):
        """Start monitoring system resources."""
        self.running = True
        while self.running:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_gb = memory.used / (1024**3)
            
            # Store measurements
            self.cpu_percent_history.append(cpu_percent)
            self.memory_percent_history.append(memory_percent)
            self.memory_gb_history.append(memory_gb)
            self.timestamps.append(time.time())
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        if not self.cpu_percent_history:
            return {}
            
        return {
            "cpu": {
                "mean": np.mean(self.cpu_percent_history),
                "max": np.max(self.cpu_percent_history),
                "std": np.std(self.cpu_percent_history)
            },
            "memory": {
                "mean_percent": np.mean(self.memory_percent_history),
                "max_percent": np.max(self.memory_percent_history),
                "mean_gb": np.mean(self.memory_gb_history),
                "max_gb": np.max(self.memory_gb_history)
            },
            "duration": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        }


class TrainingMetricsTracker:
    """Track training metrics over time."""
    
    def __init__(self):
        self.update_count = 0
        self.loss_history = []
        self.kl_history = []
        self.entropy_history = []
        self.pg_loss_history = []
        self.accuracy_history = []
        self.reward_history = []
        self.update_times = []
        self.episode_times = []
        self.timestamps = []
        
    def record_update(self, stats: Dict[str, float]):
        """Record update statistics."""
        self.update_count += 1
        self.timestamps.append(time.time())
        
        # Extract metrics
        self.loss_history.append(stats.get('loss', 0))
        self.kl_history.append(stats.get('kl', 0))
        self.entropy_history.append(stats.get('entropy', 0))
        self.pg_loss_history.append(stats.get('pg_loss', 0))
        
    def record_episode(self, reward: float, duration: float):
        """Record episode statistics."""
        self.reward_history.append(reward)
        self.episode_times.append(duration)
        
        # Calculate running accuracy (last 100 episodes)
        recent_rewards = self.reward_history[-100:]
        accuracy = sum(r > 0 for r in recent_rewards) / len(recent_rewards) if recent_rewards else 0
        self.accuracy_history.append(accuracy)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.loss_history:
            return {}
            
        return {
            "updates": self.update_count,
            "episodes": len(self.reward_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0,
            "final_accuracy": self.accuracy_history[-1] if self.accuracy_history else 0,
            "loss_reduction": (self.loss_history[0] - self.loss_history[-1]) if len(self.loss_history) > 1 else 0,
            "mean_episode_time": np.mean(self.episode_times) if self.episode_times else 0,
            "total_time": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        }
    
    def print_charts(self):
        """Print ASCII charts of metrics."""
        print("\n" + "="*80)
        print("📊 TRAINING METRICS VISUALIZATION")
        print("="*80)
        
        # Loss chart
        if self.loss_history:
            print("\n📉 Loss Over Time:")
            self._print_chart(self.loss_history, height=10)
            print(f"   Initial: {self.loss_history[0]:.4f} → Final: {self.loss_history[-1]:.4f}")
        
        # KL divergence chart
        if self.kl_history:
            print("\n📈 KL Divergence Over Time:")
            self._print_chart(self.kl_history, height=8)
            print(f"   Initial: {self.kl_history[0]:.4f} → Final: {self.kl_history[-1]:.4f}")
        
        # Accuracy chart (if we have enough data)
        if len(self.accuracy_history) > 10:
            print("\n🎯 Accuracy Over Time (%):")
            accuracy_percent = [a * 100 for a in self.accuracy_history]
            self._print_chart(accuracy_percent, height=10, is_percentage=True)
            print(f"   Initial: {accuracy_percent[0]:.1f}% → Final: {accuracy_percent[-1]:.1f}%")
        
        # Episode time distribution
        if self.episode_times:
            print("\n⏱️  Episode Time Distribution (seconds):")
            self._print_histogram(self.episode_times, bins=10)
            print(f"   Mean: {np.mean(self.episode_times):.2f}s, Std: {np.std(self.episode_times):.2f}s")
        
        print("\n" + "="*80)
    
    def _print_chart(self, data: List[float], height: int = 10, width: int = 60, is_percentage: bool = False):
        """Print a simple ASCII line chart."""
        if not data:
            return
            
        # Downsample if too many points
        if len(data) > width:
            indices = np.linspace(0, len(data)-1, width, dtype=int)
            data = [data[i] for i in indices]
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Create the chart
        for h in range(height, 0, -1):
            line = "   "
            threshold = min_val + (h / height) * range_val
            
            for val in data:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            
            # Add scale
            if is_percentage:
                label = f"{threshold:5.1f}%"
            else:
                label = f"{threshold:7.4f}"
            print(f"{label} |{line}")
        
        # X-axis
        print("         " + "└" + "─" * len(data))
        print(f"         0{' ' * (len(data)-10)}updates")
    
    def _print_histogram(self, data: List[float], bins: int = 10):
        """Print a simple ASCII histogram."""
        if not data:
            return
            
        hist, bin_edges = np.histogram(data, bins=bins)
        max_count = max(hist)
        
        for i, count in enumerate(hist):
            bar_length = int(40 * count / max_count) if max_count > 0 else 0
            bar = "█" * bar_length
            label = f"{bin_edges[i]:5.1f}-{bin_edges[i+1]:5.1f}s"
            print(f"   {label}: {bar} ({count})")


async def estimate_parallel_capacity() -> Dict[str, Any]:
    """Estimate how many environments we can run in parallel."""
    # Get system info
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024**3)
    available_memory_gb = memory.available / (1024**3)
    
    # Estimate based on typical resource usage
    # Each environment typically uses ~200-300MB RAM
    # Agent server uses ~2-4GB RAM
    # Leave 2GB for system
    agent_memory_gb = 4.0
    system_reserve_gb = 2.0
    env_memory_gb = 0.3
    
    usable_memory_gb = available_memory_gb - system_reserve_gb
    max_envs_by_memory = int((usable_memory_gb - agent_memory_gb) / env_memory_gb)
    
    # CPU-based estimate (2-4 envs per core is reasonable)
    max_envs_by_cpu = cpu_count * 3
    
    # Take the minimum
    recommended_max = min(max_envs_by_memory, max_envs_by_cpu, 32)  # Cap at 32 for stability
    
    return {
        "cpu_count": cpu_count,
        "total_memory_gb": total_memory_gb,
        "available_memory_gb": available_memory_gb,
        "max_envs_by_memory": max_envs_by_memory,
        "max_envs_by_cpu": max_envs_by_cpu,
        "recommended_max_concurrent": recommended_max,
        "reasoning": (
            f"Based on {cpu_count} CPUs and {available_memory_gb:.1f}GB available RAM. "
            f"Reserving {agent_memory_gb}GB for agent server and {system_reserve_gb}GB for system. "
            f"Each environment uses ~{env_memory_gb}GB."
        )
    }


async def run_training_test(
    num_epochs: float = 0.5,
    k_samples: int = 4,
    buffer_min: int = 32,
    batch_size: int = 8,
    max_concurrent: Optional[int] = None
) -> Dict[str, Any]:
    """Run GRPO training test with monitoring."""
    
    # Estimate parallel capacity if not specified
    if max_concurrent is None:
        capacity = await estimate_parallel_capacity()
        max_concurrent = capacity["recommended_max_concurrent"]
        print(f"\n🖥️  System Capacity Analysis:")
        print(f"   {capacity['reasoning']}")
        print(f"   Recommended max concurrent: {max_concurrent}")
    
    # Load tasks
    tasks_path = Path("examples/rl/data/math_tasks/train_tasks_large.json")
    if not tasks_path.exists():
        # Fallback to smaller dataset
        tasks_path = Path("examples/rl/data/math_tasks/train_tasks.json")
        
    with open(tasks_path) as f:
        task_configs = json.load(f)
    
    tasks = [Task.from_dict(config) for config in task_configs]
    print(f"\n📚 Loaded {len(tasks)} training tasks")
    
    # Create agent
    agent_url = os.environ.get("AGENT_URL", "http://localhost:8000")
    agent = HostedVLMAgent(
        base_url=agent_url,
        timeout=120.0,
    )
    
    # Initialize monitors
    system_monitor = SystemMonitor()
    metrics_tracker = TrainingMetricsTracker()
    
    # Hook into agent to track metrics
    original_update = agent.update
    async def monitored_update(batch):
        update_start = time.time()
        stats = await original_update(batch)
        update_time = time.time() - update_start
        
        metrics_tracker.record_update(stats)
        logger.info(
            f"Update {metrics_tracker.update_count}: "
            f"Loss={stats.get('loss', 0):.4f}, "
            f"PG={stats.get('pg_loss', 0):.4f}, "
            f"KL={stats.get('kl', 0):.4f}, "
            f"Time={update_time:.1f}s"
        )
        return stats
    
    agent.update = monitored_update
    
    # Create trainer
    trainer = GRPOTrainer(
        agent=agent,
        dataset=tasks,
        K=k_samples,
        buffer_min=buffer_min,
        batch_size=batch_size,
        max_concurrent=max_concurrent if max_concurrent is not None else 16,
        show_dashboard=True,
    )
    
    # Calculate expected updates
    updates_per_epoch = trainer.calculate_updates_per_epoch()
    total_updates = int(num_epochs * updates_per_epoch)
    
    print(f"\n📋 Training Configuration:")
    print(f"   Tasks: {len(tasks)}")
    print(f"   K samples per task: {k_samples}")
    print(f"   Buffer size: {buffer_min} trajectories")
    print(f"   Groups before update: ~{buffer_min // k_samples}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max concurrent: {max_concurrent}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Expected updates: {total_updates}")
    print(f"   Expected episodes: {int(len(tasks) * k_samples * num_epochs)}")
    
    # Start monitoring
    monitor_task = asyncio.create_task(system_monitor.start_monitoring())
    
    print(f"\n🚀 Starting GRPO training...\n")
    
    try:
        # Run training
        start_time = time.time()
        final_stats = await trainer.train(num_epochs=num_epochs)
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.1f} seconds")
        
    finally:
        # Stop monitoring
        system_monitor.stop_monitoring()
        await asyncio.sleep(2)  # Let monitor finish
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    # Get summaries
    system_summary = system_monitor.get_summary()
    metrics_summary = metrics_tracker.get_summary()
    
    # Print results
    print("\n" + "="*80)
    print("📊 TRAINING RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   Total time: {training_time:.1f}s")
    print(f"   Episodes completed: {metrics_summary.get('episodes', 0)}")
    print(f"   Updates completed: {metrics_summary.get('updates', 0)}")
    print(f"   Episodes/second: {metrics_summary.get('episodes', 0) / training_time:.2f}")
    print(f"   Mean episode time: {metrics_summary.get('mean_episode_time', 0):.2f}s")
    
    print(f"\n🖥️  System Resource Usage:")
    print(f"   CPU: {system_summary['cpu']['mean']:.1f}% avg, {system_summary['cpu']['max']:.1f}% max")
    print(f"   Memory: {system_summary['memory']['mean_gb']:.1f}GB avg, {system_summary['memory']['max_gb']:.1f}GB max")
    
    print(f"\n📈 Training Progress:")
    print(f"   Final loss: {metrics_summary.get('final_loss', 0):.4f}")
    print(f"   Loss reduction: {metrics_summary.get('loss_reduction', 0):.4f}")
    print(f"   Final accuracy: {metrics_summary.get('final_accuracy', 0)*100:.1f}%")
    
    # Print efficiency analysis
    if 'parallelization' in trainer.stats_tracker.get_summary():
        para_stats = trainer.stats_tracker.get_summary()['parallelization']
        print(f"\n🔧 Parallelization Efficiency:")
        print(f"   Worker utilization: {para_stats['avg_utilization']*100:.1f}%")
        print(f"   Parallel efficiency: {para_stats['parallel_efficiency']*100:.1f}%")
    
    # Print charts
    metrics_tracker.print_charts()
    
    return {
        "training_time": training_time,
        "system_stats": system_summary,
        "metrics_stats": metrics_summary,
        "final_stats": final_stats,
        "config": {
            "num_epochs": num_epochs,
            "k_samples": k_samples,
            "buffer_min": buffer_min,
            "batch_size": batch_size,
            "max_concurrent": max_concurrent,
            "num_tasks": len(tasks)
        }
    }


async def main():
    """Run performance test with different configurations."""
    # Test with recommended settings
    results = await run_training_test(
        num_epochs=0.25,  # Start with quarter epoch for testing
        k_samples=8,      # 8 samples per task
        buffer_min=512,   # 512 total trajectories before update (64 groups of 8)
        batch_size=64,    # Larger batch size for efficiency
        max_concurrent=None  # Auto-detect
    )
    
    # Save results
    results_dir = Path("examples/rl/results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"grpo_performance_{timestamp}.json"
    
    with open(results_file, "w") as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Recommendations for larger test
    print("\n" + "="*80)
    print("🎯 RECOMMENDATIONS FOR LARGER TEST")
    print("="*80)
    
    efficiency = results['metrics_stats'].get('episodes', 0) / results['training_time'] if results['training_time'] > 0 else 0
    
    print(f"\nBased on this test run:")
    print(f"   - Episodes/second: {efficiency:.2f}")
    print(f"   - System can handle: {results['config']['max_concurrent']} concurrent environments")
    
    if efficiency > 1.0:
        print(f"\n✅ Good efficiency! For a larger test, consider:")
        print(f"   - Epochs: 1.0-2.0")
        print(f"   - Buffer size: 1024-2048 (total trajectories)")
        print(f"   - Batch size: 64-128")
        print(f"   - K samples: 8-16")
    else:
        print(f"\n⚠️  Lower efficiency detected. For larger test:")
        print(f"   - Consider reducing max_concurrent to {max(8, results['config']['max_concurrent']//2)}")
        print(f"   - Keep buffer_min at 256-512")
        print(f"   - Use smaller batch_size: 32-64")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main()) 