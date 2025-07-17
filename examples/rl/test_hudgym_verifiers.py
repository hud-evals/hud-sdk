"""
Test HUDGym environment with verifiers.env.evaluate.
"""

import os
import json
from typing import Optional
import asyncio
from openai import OpenAI

from verifiers_demo import HUDGym
from hud.task import Task
from hud.adapters.common.adapter import Adapter

class BasicAdapter(Adapter):
    """Adapter that extracts actions from model output."""
    
    def preprocess(self, model_output: str) -> str:
        """Extract action from model output."""
        # Look for action tags
        if "<action>" in model_output and "</action>" in model_output:
            start = model_output.find("<action>") + 8
            end = model_output.find("</action>")
            action = model_output[start:end].strip()
            return action
        
        # Look for basic commands
        lines = model_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if any(cmd in line.lower() for cmd in ['click', 'type', 'scroll', 'message', 'done']):
                return line
        
        # Default: return the whole output
        return model_output.strip()
    
    def convert(self, action_str: str):
        """Convert action string to CLA format."""
        # Parse basic action commands
        action_str = action_str.strip().lower()
        
        if action_str.startswith('click'):
            # Click action
            return {"type": "click", "point": {"x": 100, "y": 100}}
        elif action_str.startswith('type'):
            # Extract text to type
            if '"' in action_str:
                text = action_str.split('"')[1]
                return {"type": "type", "text": text}
            else:
                return {"type": "type", "text": ""}
        elif action_str.startswith('done'):
            # Extract response text if provided
            if '"' in action_str:
                text = action_str.split('"')[1]
                return {"type": "response", "text": text}
            else:
                return {"type": "response", "text": "Task completed"}
        else:
            # if no recognized action, we need to end the task
            return {}  # will throw an error

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
    tasks_file = os.path.join(script_dir, "data", "math_tasks", "train_tasks_large_2.json")
    with open(tasks_file, 'r') as f:
        tasks_data = json.load(f)
    
    print(f"Testing {len(tasks_data)} tasks concurrently...")
    
    # Create all tasks
    tasks = [Task.from_dict(task_data) for task_data in tasks_data]

    # Create adapter
    adapter = BasicAdapter()

    # Create HUDGym environment with all tasks
    env = HUDGym(
        tasks=tasks,
        adapter=adapter,
        client=client,
        model=model,
        max_steps=5
    )
    
    try:
        print(f"Testing HUDGym environment evaluation with {model}...")
        
        # Run evaluation
        sampling_args = {
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        
        results = env.evaluate(
            sampling_args=sampling_args,
            num_examples=len(tasks_data),
            rollouts_per_example=1
        )
        
        print("\n--- HUDGym Evaluation Results ---")
        print(f"Number of examples: {len(results.get('prompt', []))}")
        
        if results.get('prompt'):
            print(f"\nExample:")
            prompt = results['prompt'][0]
            if isinstance(prompt, list) and len(prompt) > 1:
                print(f"Task prompt: {prompt[1]['content'][:100]}...")
            
            completion = results['completion'][0]
            if isinstance(completion, list) and completion:
                print(f"Model response: {completion[-1].get('content', '')[:200]}...")
            
            print(f"Reward: {results['reward'][0]}")
        
        print("\n--- Reward Breakdown ---")
        for key, value in results.items():
            if 'reward' in key.lower() and isinstance(value, list) and value:
                avg_reward = sum(value) / len(value)
                print(f"{key}: {avg_reward:.3f}")
        
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