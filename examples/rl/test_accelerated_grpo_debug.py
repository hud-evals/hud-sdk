#!/usr/bin/env python3
"""Debug version of GRPO test to diagnose training issues."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from hud.agent.accelerated_vlm import AcceleratedVLMAgent
from hud.rl import GRPOTrainer
from hud.task import Task

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also set debug for key modules
logging.getLogger('hud.rl').setLevel(logging.DEBUG)
logging.getLogger('hud.agent').setLevel(logging.DEBUG)


async def run_grpo_training_debug(
    num_epochs: float = 0.1,
    k_samples: int = 2,
    buffer_min: int = 8,
    batch_size: int = 4,
    max_concurrent: int = 1,
):
    """Run GRPO training with debug logging."""
    
    # Load tasks
    tasks_path = Path("examples/rl/data/math_tasks/train_tasks.json")
    if not tasks_path.exists():
        logger.error(f"Tasks file not found: {tasks_path}")
        return
        
    with open(tasks_path) as f:
        task_configs = json.load(f)
    
    # Use just 5 tasks for debugging
    tasks = [Task.from_dict(config) for config in task_configs[:5]]
    logger.info(f"📚 Loaded {len(tasks)} training tasks")
    
    # Log task details
    for i, task in enumerate(tasks):
        logger.debug(f"Task {i}: {task.id} - {task.prompt[:50]}...")
    
    # Create AcceleratedVLMAgent with smaller model
    logger.info("Creating AcceleratedVLMAgent...")
    agent = AcceleratedVLMAgent(
        gradient_accumulation_steps=1,
        mixed_precision="no",  # Disable for debugging
        inference_threads=1,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model
        device_map="auto",
        load_in_8bit=False,
        use_lora=False,
        learning_rate=1e-5,
        max_new_tokens=100,
        temperature=0.7,
        system_prompt=(
            "You are a helpful math tutor. Solve the given math problem step by step. "
            "Show your work clearly and provide the final answer."
        )
    )
    
    logger.info(f"🤖 Created AcceleratedVLMAgent")
    
    # Test the agent with a simple sample before training
    logger.info("Testing agent with a simple sample...")
    from hud.utils.common import Observation
    test_obs = Observation(text="What is 2+2?")
    try:
        sample = await agent.sample(test_obs)
        logger.info(f"Test successful! Generated: {sample.text[:100]}...")
    except Exception as e:
        logger.error(f"Agent test failed: {e}", exc_info=True)
        return
    
    # Create GRPO trainer with debug logging disabled
    trainer = GRPOTrainer(
        agent=agent,
        dataset=tasks,
        K=k_samples,
        buffer_min=buffer_min,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        show_dashboard=False,  # Disable dashboard for debug output
    )
    
    logger.info(f"\n📋 Training Configuration:")
    logger.info(f"   Tasks: {len(tasks)}")
    logger.info(f"   K samples per task: {k_samples}")
    logger.info(f"   Buffer size: {buffer_min}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max concurrent: {max_concurrent}")
    logger.info(f"   Epochs: {num_epochs}")
    
    logger.info(f"\n🚀 Starting GRPO training...")
    
    # Run training
    try:
        final_stats = await trainer.train(num_epochs=num_epochs)
        
        logger.info("Training completed!")
        logger.info(f"Final stats: {final_stats}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)


async def main():
    """Main entry point."""
    await run_grpo_training_debug()


if __name__ == "__main__":
    asyncio.run(main()) 