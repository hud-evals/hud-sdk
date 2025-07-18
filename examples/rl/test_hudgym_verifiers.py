"""
Test HUDGym environment with verifiers.env.evaluate and statistics tracking.
"""

import os
import json
from typing import Optional
import asyncio
from openai import OpenAI

from verifiers_demo import HUDGym, BasicAdapter
from hud.task import Task
import hud.gym as gym

async def test_hudgym_evaluate(base_url: Optional[str] = None):
    """Test HUDGym environment evaluation."""
    
    # Set up OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not base_url:
        print("Please set OPENAI_API_KEY environment variable or provide --base-url for a vLLM server.")
        return

    # For vLLM, api_key is can be a dummy value.
    if base_url and not api_key:
        api_key = "EMPTY"
    
    effective_base_url = base_url or "https://api.openai.com/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=effective_base_url
    )

    if base_url:
        models = client.models.list()
        model = models.data[0].id
        print(f"Using model: {model} from vLLM server at {base_url}")
    else:
        model = "gpt-4.1-mini"  # Default model for OpenAI API
        print(f"Using model: {model} with OpenAI API")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_file = os.path.join(script_dir, "data", "math_tasks", "test_tasks_large_2.json")
    with open(tasks_file, 'r') as f:
        tasks_data = json.load(f)
    
    print(f"Testing {len(tasks_data)} tasks concurrently...")
    
    # Create all tasks
    tasks = [Task.from_dict(task_data) for task_data in tasks_data]

    # Create adapter
    adapter = BasicAdapter()

    # Create HUDGym environment with all tasks and stats enabled
    env = HUDGym(
        tasks=tasks[:1],
        adapter=adapter,
        max_turns=5,
        enable_stats=True  # Enable statistics tracking
    )

    try:
        print(f"Testing HUDGym environment evaluation with {model}...")
        
        # Run evaluation
        sampling_args = {
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        
        results = env.evaluate(
            client=client,
            model=model,
            sampling_args=sampling_args,
            num_examples=len(tasks_data),
            rollouts_per_example=1,
            max_concurrent_rollouts=10,
        )
        
        print("\n--- HUDGym Evaluation Results ---")
        print(f"Number of examples: {len(results.get('prompt', []))}")
        
        if results.get('prompt'):
            print(f"\nExample:")
            prompt = results['prompt'][0]
            if isinstance(prompt, list) and len(prompt) > 1:
                print(f"Task prompt: {prompt[1]['content']}...")
            
            completion = results['completion'][0]
            if isinstance(completion, list) and completion:
                print(f"Model response: {completion[-1].get('content', '')}...")
            
            print(f"Reward: {results['reward'][0]}")
        
        print("\n--- Reward Breakdown ---")
        for key, value in results.items():
            if 'reward' in key.lower() and isinstance(value, list) and value:
                avg_reward = sum(value) / len(value)
                print(f"{key}: {avg_reward:.3f}")

        # # Optional: make dataset
        # test_dataset = env.make_dataset(results)
        # test_dataset = test_dataset.sort("reward").select(range(10))
        # print("\n---Bottom 5 Test Dataset Samples ---")
        # for i, sample in enumerate(test_dataset):
        #     print(f"Sample {i+1}:")
        #     print(f"  Task: {sample['task']}")
        #     print(f"  Prompt: {sample['prompt'][:100]}...")
        #     print(f"  Completion: {sample['completion'][:200]}...")
        #     print(f"  Reward: {sample['reward']:.3f}")

        # Display statistics summary
        print(env.get_stats_summary())

        print("\nHUDGym evaluation test completed successfully!")
        
    except Exception as e:
        print(f"Error during HUDGym evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate HUDGym using OpenAI API or a OpenAI compatible server.")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the OpenAI API compatible vLLM server.")
    args = parser.parse_args()

    asyncio.run(test_hudgym_evaluate(base_url=args.base_url))