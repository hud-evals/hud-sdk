#!/usr/bin/env python3
"""
Test HUDGym environment with verifiers.env.evaluate on gpt-4.1-mini.
"""

import os
import asyncio
from openai import AsyncOpenAI, OpenAI

# Add examples/rl to path for verifiers_demo
import sys
sys.path.insert(0, '/Users/jaideepchawla/dev/hud/hud-sdk/examples/rl')

from verifiers_demo import HUDGym
from hud.task import Task
from hud.adapters.common.adapter import Adapter


class SimpleAdapter(Adapter):
    """Simple adapter that extracts actions from model output."""
    
    def preprocess(self, model_output: str) -> str:
        """Extract action from model output."""
        # Look for action tags
        if "<action>" in model_output and "</action>" in model_output:
            start = model_output.find("<action>") + 8
            end = model_output.find("</action>")
            action = model_output[start:end].strip()
            return action
        
        # Look for simple commands
        lines = model_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if any(cmd in line.lower() for cmd in ['click', 'type', 'scroll', 'message', 'done']):
                return line
        
        # Default: return the whole output
        return model_output.strip()
    
    def convert(self, action_str: str):
        """Convert action string to CLA format."""
        # Parse simple action commands
        action_str = action_str.strip().lower()
        
        if action_str.startswith('click'):
            # Basic click action
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
            # Default action - return response
            return {"type": "response", "text": action_str}


async def test_hudgym_evaluate():
    """Test HUDGym environment evaluation with gpt-4.1-mini."""
    
    # Set up OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create sync client for verifiers
    client = OpenAI(api_key=api_key)
    
    # Load all tasks from JSON  
    import json
    with open('/Users/jaideepchawla/dev/hud/hud-sdk/examples/rl/data/math_tasks/train_tasks_small.json', 'r') as f:
        tasks_data = json.load(f)
    
    print(f"Testing {len(tasks_data)} tasks concurrently...")
    
    # Use the first task to create the environment
    task_data = tasks_data[0]
    task = Task.from_dict(task_data)
    
    # Create adapter
    adapter = SimpleAdapter()
    
    # Create HUDGym environment
    env = HUDGym(
        task=task,
        adapter=adapter,
        client=client,
        model="gpt-4.1-mini",
        max_steps=5
    )
    
    try:
        print("Testing HUDGym environment evaluation with gpt-4.1-mini...")
        
        # Run evaluation
        sampling_args = {
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        
        results = env.evaluate(
            client=client,
            model="gpt-4.1-mini",
            sampling_args=sampling_args,
            num_examples=1,
            rollouts_per_example=len(tasks_data)
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
    asyncio.run(test_hudgym_evaluate())