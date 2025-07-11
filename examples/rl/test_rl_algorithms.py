#!/usr/bin/env python3
"""Test RL algorithms (GRPO and DAPO) with HostedVLMAgent.

This script demonstrates both algorithms:
- GRPO: Group Relative Policy Optimization (DeepSeek-R1 style)
- DAPO: Diversity-Aware Policy Optimization (with clip-higher and dynamic sampling)

Usage:
    # Test GRPO (default)
    python test_rl_algorithms.py
    
    # Test DAPO
    python test_rl_algorithms.py --algorithm dapo
    
    # With custom parameters
    python test_rl_algorithms.py --algorithm dapo --k-samples 8 --epochs 0.5
    
    # With remote server
    AGENT_URL=http://54.123.45.67:8000 python test_rl_algorithms.py --algorithm grpo
"""

import asyncio
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

from hud.agent.hosted_vlm_agent import HostedVLMAgent
from hud.rl import GRPOTrainer, DAPOTrainer
from hud.task import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLMonitor:
    """Monitor RL training progress and statistics."""
    
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.update_count = 0
        self.update_stats = []
        self.group_stats = []
        self.start_time = datetime.now()
        
    def record_update(self, stats: Dict[str, float], batch_info: Dict[str, Any]):
        """Record update statistics."""
        self.update_count += 1
        self.update_stats.append({
            "update": self.update_count,
            "stats": stats,
            "batch_info": batch_info,
            "timestamp": datetime.now().isoformat()
        })
        
    def record_group(self, task_id: str, rewards: List[float], advantages: List[float]):
        """Record group completion statistics."""
        self.group_stats.append({
            "task_id": task_id,
            "rewards": rewards,
            "advantages": advantages,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "timestamp": datetime.now().isoformat()
        })
        
    def print_summary(self):
        """Print training summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"🎯 {self.algorithm} Training Summary")
        print(f"{'='*60}")
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total updates: {self.update_count}")
        print(f"   Total groups: {len(self.group_stats)}")
        print(f"   Training time: {duration:.1f}s")
        
        if self.update_stats:
            # Loss progression
            losses = [u['stats']['loss'] for u in self.update_stats]
            print(f"\n📈 Loss Progression:")
            print(f"   Initial loss: {losses[0]:.4f}")
            print(f"   Final loss: {losses[-1]:.4f}")
            print(f"   Change: {losses[-1] - losses[0]:+.4f}")
            
            # Algorithm-specific metrics
            if self.algorithm == "GRPO":
                # GRPO metrics
                pg_losses = [u['stats']['pg_loss'] for u in self.update_stats]
                kls = [u['stats']['kl'] for u in self.update_stats]
                
                print(f"\n🔄 GRPO Metrics:")
                print(f"   Policy gradient loss: {pg_losses[0]:.4f} → {pg_losses[-1]:.4f}")
                print(f"   KL divergence: {kls[0]:.4f} → {kls[-1]:.4f}")
                
            elif self.algorithm == "DAPO":
                # DAPO metrics
                if 'avg_weight' in self.update_stats[-1]['stats']:
                    weights = [u['stats'].get('avg_weight', 1.0) for u in self.update_stats]
                    clip_fracs = [u['stats'].get('clip_fraction', 0) for u in self.update_stats]
                    
                    print(f"\n🎭 DAPO Metrics:")
                    print(f"   Avg dynamic weight: {weights[0]:.4f} → {weights[-1]:.4f}")
                    print(f"   Clip fraction: {clip_fracs[0]:.3f} → {clip_fracs[-1]:.3f}")
        
        if self.group_stats:
            # Task performance
            all_rewards = [r for g in self.group_stats for r in g['rewards']]
            print(f"\n🎯 Task Performance:")
            print(f"   Total samples: {len(all_rewards)}")
            print(f"   Success rate: {sum(r > 0 for r in all_rewards) / len(all_rewards):.1%}")
            print(f"   Mean reward: {np.mean(all_rewards):.3f}")
            print(f"   Reward std: {np.std(all_rewards):.3f}")
            
            # Advantage distribution
            all_advantages = [a for g in self.group_stats for a in g['advantages']]
            print(f"\n📊 Advantage Distribution:")
            print(f"   Mean: {np.mean(all_advantages):.4f}")
            print(f"   Std: {np.std(all_advantages):.4f}")
            print(f"   Min: {np.min(all_advantages):.4f}")
            print(f"   Max: {np.max(all_advantages):.4f}")
            
            # Groups with zero advantages (all same reward)
            zero_adv_groups = sum(1 for g in self.group_stats if np.std(g['rewards']) < 1e-6)
            print(f"   Zero-advantage groups: {zero_adv_groups}/{len(self.group_stats)} ({zero_adv_groups/len(self.group_stats):.1%})")
        
        print(f"\n{'='*60}\n")


async def main():
    """Test RL algorithms with configurable parameters."""
    parser = argparse.ArgumentParser(description="Test RL algorithms (GRPO/DAPO)")
    parser.add_argument(
        "--algorithm", 
        choices=["grpo", "dapo"], 
        default="grpo",
        help="RL algorithm to test (default: grpo)"
    )
    parser.add_argument(
        "--k-samples", 
        type=int, 
        default=4,
        help="Number of samples per task (default: 4)"
    )
    parser.add_argument(
        "--buffer-min", 
        type=int, 
        default=16,
        help="Minimum buffer size before update (default: 16)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="Batch size for updates (default: 8)"
    )
    parser.add_argument(
        "--epochs", 
        type=float, 
        default=0.5,
        help="Number of epochs to train (default: 0.5)"
    )
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        default=4,
        help="Maximum concurrent tasks (default: 4)"
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        default="examples/rl/data/math_tasks/test_tasks.json",
        help="Path to tasks JSON file"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=12,
        help="Number of tasks to use from the file (default: 12)"
    )
    
    args = parser.parse_args()
    
    # Convert algorithm to uppercase for display
    algorithm = args.algorithm.upper()
    
    print(f"\n🚀 Testing {algorithm} Implementation\n")
    print(f"📋 Configuration:")
    print(f"   Algorithm: {algorithm}")
    print(f"   K samples per task: {args.k_samples}")
    print(f"   Buffer size: {args.buffer_min}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Max concurrent: {args.max_concurrent}")
    
    # Load tasks
    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        logger.error(f"Tasks file not found: {tasks_path}")
        return
        
    with open(tasks_path) as f:
        task_configs = json.load(f)[:args.num_tasks]
    
    tasks = [Task.from_dict(config) for config in task_configs]
    print(f"\n📚 Loaded {len(tasks)} tasks")
    
    # Get agent URL from environment or use default
    agent_url = os.environ.get("AGENT_URL", "http://localhost:8000")
    print(f"\n🌐 Using agent at: {agent_url}")
    
    # Create agent and monitor
    agent = HostedVLMAgent(
        base_url=agent_url,
        timeout=120.0,
    )
    
    monitor = RLMonitor(algorithm)
    
    # Hook into agent's update method to track statistics
    original_update = agent.update
    async def monitored_update(batch):
        # Track batch info
        batch_info = {
            'batch_size': len(batch.texts),
            'has_old_log_probs': batch.old_log_probs is not None,
            'algorithm': batch.metadata.get('algorithm', algorithm) if batch.metadata else algorithm,
        }
        
        # Perform update
        stats = await original_update(batch)
        
        # Record statistics
        monitor.record_update(stats, batch_info)
        
        # Log key metrics based on algorithm
        if algorithm == "GRPO":
            logger.info(
                f"Update {monitor.update_count}: "
                f"Loss={stats['loss']:.4f}, "
                f"PG={stats.get('pg_loss', 0):.4f}, "
                f"KL={stats.get('kl', 0):.4f}"
            )
        else:  # DAPO
            logger.info(
                f"Update {monitor.update_count}: "
                f"Loss={stats['loss']:.4f}, "
                f"PG={stats.get('pg_loss', 0):.4f}, "
                f"Clip={stats.get('clip_fraction', 0):.3f}, "
                f"Weight={stats.get('avg_weight', 1.0):.3f}"
            )
        
        return stats
    
    agent.update = monitored_update
    
    # Create appropriate trainer
    if args.algorithm == "grpo":
        trainer = GRPOTrainer(
            agent=agent,
            dataset=tasks,
            K=args.k_samples,
            buffer_min=args.buffer_min,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            show_dashboard=True,
        )
    else:  # DAPO
        trainer = DAPOTrainer(
            agent=agent,
            dataset=tasks,
            K=args.k_samples,
            buffer_min=args.buffer_min,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            show_dashboard=True,
            # DAPO uses lambda_temp for dynamic sampling as per the paper
            lambda_temp=2.0,  # Temperature for dynamic weight calculation
        )
    
    # Train
    print(f"\n🏃 Starting {algorithm} training for {args.epochs} epochs...\n")
    
    try:
        await trainer.train(num_epochs=args.epochs)
        print(f"\n✅ {algorithm} training completed successfully!")
    except KeyboardInterrupt:
        print(f"\n⚠️  {algorithm} training interrupted by user")
    except Exception as e:
        print(f"\n❌ {algorithm} training failed with error: {e}")
        logger.exception("Training error")
    
    # Print summary
    monitor.print_summary()
    
    # Save detailed results
    results_dir = Path("examples/rl/results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{args.algorithm}_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "algorithm": algorithm,
            "config": vars(args),
            "update_stats": monitor.update_stats,
            "group_stats": monitor.group_stats,
            "duration": (datetime.now() - monitor.start_time).total_seconds(),
        }, f, indent=2)
    
    print(f"📊 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main()) 