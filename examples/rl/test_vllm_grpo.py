#!/usr/bin/env python3
"""Test GRPO training with vLLM server using HybridVLMAgent."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from hud.agent.hybrid_vlm import HybridVLMAgent
from hud.rl import GRPOTrainer
from hud.task import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_vllm_server():
    """Test if vLLM server is running and accessible."""
    import httpx
    
    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    try:
        async with httpx.AsyncClient() as client:
            # Check health endpoint
            resp = await client.get(f"{vllm_url}/health")
            if resp.status_code == 200:
                logger.info("✅ vLLM server is healthy")
            
            # Check models endpoint
            resp = await client.get(f"{vllm_url}/v1/models")
            if resp.status_code == 200:
                models = resp.json()
                logger.info(f"📋 Available models: {models}")
                return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to vLLM server: {e}")
        return False
    
    return False


async def run_grpo_training(
    num_epochs: float = 0.25,
    k_samples: int = 4,
    buffer_min: int = 32,
    batch_size: int = 8,
    max_concurrent: int = 16,
    vllm_url: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """Run GRPO training with vLLM server."""
    
    # Check vLLM server
    if not await test_vllm_server():
        logger.error("Please start vLLM server first with:")
        logger.error("python -m vllm.entrypoints.openai.api_server \\")
        logger.error("    --model Qwen/Qwen2.5-7B-Instruct \\")
        logger.error("    --dtype float16 \\")
        logger.error("    --max-model-len 2048 \\")
        logger.error("    --gpu-memory-utilization 0.9")
        return
    
    # Load tasks
    tasks_path = Path("examples/rl/data/math_tasks/train_tasks.json")
    if not tasks_path.exists():
        logger.error(f"Tasks file not found: {tasks_path}")
        return
        
    with open(tasks_path) as f:
        task_configs = json.load(f)
    
    tasks = [Task.from_dict(config) for config in task_configs[:100]]  # Use subset for testing
    logger.info(f"📚 Loaded {len(tasks)} training tasks")
    
    # Create HybridVLMAgent
    vllm_url = vllm_url or os.environ.get("VLLM_URL", "http://localhost:8000")
    model_name = model_name or os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    
    agent = HybridVLMAgent(
        # vLLM config for fast inference
        vllm_url=vllm_url,
        vllm_timeout=30.0,
        # Local model config for gradient updates
        model_name=model_name,
        device_map="auto",
        load_in_8bit=True,  # Use 8-bit quantization to save memory
        use_lora=True,      # Use LoRA for efficient training
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
    
    logger.info(f"🤖 Created HybridVLMAgent")
    logger.info(f"   vLLM Server: {vllm_url} (for fast inference)")
    logger.info(f"   Local Model: {model_name} (for gradient updates)")
    logger.info(f"   Using LoRA: rank=16, alpha=32")
    
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
    
    logger.info(f"\n🚀 Starting GRPO training with hybrid approach...")
    logger.info(f"   Inference: vLLM (fast)")
    logger.info(f"   Updates: Local model with LoRA")
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
    
    # Clean up
    await agent.aclose()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GRPO training with vLLM")
    parser.add_argument("--epochs", type=float, default=0.25, help="Number of epochs")
    parser.add_argument("--k-samples", type=int, default=4, help="K samples per task")
    parser.add_argument("--buffer-min", type=int, default=32, help="Minimum buffer size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent environments")
    parser.add_argument("--vllm-url", type=str, help="vLLM server URL")
    parser.add_argument("--model", type=str, help="Model name")
    
    args = parser.parse_args()
    
    await run_grpo_training(
        num_epochs=args.epochs,
        k_samples=args.k_samples,
        buffer_min=args.buffer_min,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        vllm_url=args.vllm_url,
        model_name=args.model,
    )


if __name__ == "__main__":
    asyncio.run(main()) 