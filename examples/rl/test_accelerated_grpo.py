#!/usr/bin/env python3
"""Test GRPO training with AcceleratedVLMAgent for distributed/faster training.

To run with Accelerate on multiple GPUs:
    accelerate launch --num_processes 4 test_accelerated_grpo.py

To run on single GPU:
    python test_accelerated_grpo.py
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from hud.agent.accelerated_vlm import AcceleratedVLMAgent
from hud.rl import GRPOTrainer
from hud.task import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_grpo_training(
    num_epochs: float = 0.25,
    k_samples: int = 4,
    buffer_min: int = 64,
    batch_size: int = 16,
    max_concurrent: int = 8,  # Lower since we're doing more work per GPU
    model_name: Optional[str] = None,
):
    """Run GRPO training with AcceleratedVLMAgent."""
    
    # Load tasks
    tasks_path = Path("examples/rl/data/math_tasks/train_tasks.json")
    if not tasks_path.exists():
        logger.error(f"Tasks file not found: {tasks_path}")
        return
        
    with open(tasks_path) as f:
        task_configs = json.load(f)
    
    tasks = [Task.from_dict(config) for config in task_configs[:100]]  # Use subset for testing
    logger.info(f"📚 Loaded {len(tasks)} training tasks")
    
    # Create AcceleratedVLMAgent
    model_name = model_name or "Qwen/Qwen2.5-7B-Instruct"
    
    agent = AcceleratedVLMAgent(
        # Accelerate config
        gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch
        mixed_precision="fp16",  # Use FP16 for faster training
        inference_threads=2,  # Multiple threads for async inference
        # Model config
        model_name=model_name,
        device_map="auto",
        load_in_8bit=True,  # 8-bit quantization
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        learning_rate=1e-5,
        max_new_tokens=512,
        temperature=0.7,
        system_prompt=(
            "You are a helpful math tutor. Solve the given math problem step by step. "
            "Show your work clearly and provide the final answer."
        )
    )
    
    logger.info(f"🤖 Created AcceleratedVLMAgent")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Mixed Precision: FP16")
    logger.info(f"   Gradient Accumulation: 2 steps")
    
    # Create GRPO trainer
    trainer = GRPOTrainer(
        agent=agent,
        dataset=tasks,
        K=k_samples,
        buffer_min=buffer_min,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        show_dashboard=True,
    )
    
    # Calculate expected updates
    updates_per_epoch = trainer.calculate_updates_per_epoch()
    total_updates = int(num_epochs * updates_per_epoch)
    
    logger.info(f"\n📋 Training Configuration:")
    logger.info(f"   Tasks: {len(tasks)}")
    logger.info(f"   K samples per task: {k_samples}")
    logger.info(f"   Buffer size: {buffer_min}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max concurrent: {max_concurrent}")
    logger.info(f"   Epochs: {num_epochs}")
    logger.info(f"   Expected updates: {total_updates}")
    
    logger.info(f"\n🚀 Starting GRPO training with Accelerate...")
    logger.info(f"   Inference: Optimized with FP16 and thread pool")
    logger.info(f"   Updates: Distributed across available GPUs")
    logger.info("")
    
    # Run training
    final_stats = await trainer.train(num_epochs=num_epochs)
    
    # Print final statistics
    print("\n" + "="*80)
    print("📊 TRAINING COMPLETED")
    print("="*80)
    print(f"\nFinal statistics:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Save checkpoint
    if agent.accelerator.is_main_process:
        checkpoint_path = "checkpoints/grpo_accelerated"
        agent.save_checkpoint(checkpoint_path)
        print(f"\n💾 Saved checkpoint to: {checkpoint_path}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GRPO training with Accelerate")
    parser.add_argument("--epochs", type=float, default=0.25, help="Number of epochs")
    parser.add_argument("--k-samples", type=int, default=4, help="K samples per task")
    parser.add_argument("--buffer-min", type=int, default=64, help="Minimum buffer size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent environments")
    parser.add_argument("--model", type=str, help="Model name")
    
    args = parser.parse_args()
    
    await run_grpo_training(
        num_epochs=args.epochs,
        k_samples=args.k_samples,
        buffer_min=args.buffer_min,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        model_name=args.model,
    )


if __name__ == "__main__":
    asyncio.run(main()) 