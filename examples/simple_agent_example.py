#!/usr/bin/env python3
"""
Example of creating a custom agent with the HUD-SO SDK.

This example demonstrates how to:
1. Create a custom agent that outputs actions
2. Create a custom adapter that converts actions to CLA format
3. Run the agent in a loop to interact with a task
"""

import asyncio
import os
import sys
from typing import Any, Tuple

# Add the parent directory to the path (if running from examples/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hud import HUDClient
from hud.adapters.common import Adapter
from hud.adapters.common.types import ClickAction, Point, TypeAction
from hud.settings import settings


# Simple custom agent that responds to tasks
class SimpleAgent:
    def __init__(self):
        self.messages = []
        
    async def predict(self, screenshot, text):
        # This would be where your agent's logic goes
        # For this example, we'll return a simple action based on the text
        if "type" in text.lower():
            return {"action": "type_text", "text": "Hello, world!"}
        else:
            # Default: click in the center of the screen
            return {"action": "click_point", "x": 500, "y": 500}
    
    def process_response(self, response) -> Tuple[bool, Any]:
        # Convert the agent's response to an action
        # Return (is_final_response, action)
        if "action" not in response:
            return True, "I don't know how to help with that."
            
        action_type = response.get("action")
        
        if action_type == "type_text":
            return False, {"action": "type", "text": response.get("text", "")}
        elif action_type == "click_point":
            return False, {"action": "left_click", "coordinate": [response.get("x", 0), response.get("y", 0)]}
        else:
            return True, f"I don't understand the action: {action_type}"


# Custom adapter that converts agent actions to CLA format
class SimpleAdapter(Adapter):
    def __init__(self):
        super().__init__()
        self.agent_width = 1024
        self.agent_height = 768
        
    def convert(self, data: Any) -> Any:
        # Convert the action dict to CLA format
        action_type = data.get("action")
        
        if action_type == "type":
            return TypeAction(text=data.get("text", ""), enter_after=False)
            
        elif action_type == "left_click":
            coord = data.get("coordinate", [0, 0])
            return ClickAction(point=Point(x=coord[0], y=coord[1]), button="left")
            
        # Handle other actions...
        
        return super().convert(data)  # Fall back to parent's implementation


async def main():
    # Initialize client with API key from settings
    api_key = settings.api_key
    if api_key is None:
        print("Error: HUD_API_KEY environment variable is not set")
        return

    # Initialize client
    client = HUDClient(api_key=api_key)
    print("Initialized HUD client")
    
    # Load gym and evalset
    gym = await client.load_gym(id="OSWorld-Ubuntu")
    evalset = await client.load_evalset(id="OSWorld-Ubuntu")
    print(f"Loaded gym and evalset: {gym.id}")
    
    # Create the run
    run = client.create_run(name="simple-agent-run", gym=gym, evalset=evalset)
    tasks = await run.fetch_task_ids()
    print(f"Found {len(tasks)} tasks")
    
    # Initialize the agent and adapter
    agent = SimpleAgent()
    adapter = SimpleAdapter()
    
    # Create environment with the adapter
    env = await run.make(metadata={"agent_id": "simple-agent"})
    print("Created environment, waiting for it to be ready...")
    
    # Wait for environment to be ready
    while True:
        state = await env.get_env_state()
        print(f"Environment state: {state}")
        if state == "running" or state == "error":
            break
        await asyncio.sleep(5)
    
    # Only proceed if the environment is running
    if state == "running" and tasks:
        # Reset to a task (using the first task for this example)
        task_id = tasks[0]
        obs = await env.reset(task_id, metadata={"run": "simple-agent-run"})
        print(f"Task description: {obs.text}")
        
        # Agent interaction loop
        max_steps = 5
        print(f"Running agent for up to {max_steps} steps")
        
        for i in range(max_steps):
            print(f"Step {i+1}/{max_steps}")
            # Rescale screenshot
            screenshot = adapter.rescale(obs.screenshot)
            
            # Get agent's prediction
            response = await agent.predict(screenshot, obs.text)
            print(f"Agent response: {response}")
            
            # Process the response
            done, action = agent.process_response(response)
            print(f"Processed action: {action}")
            
            if done:
                # This is a final response
                env.final_response = str(action)
                print(f"Final response: {env.final_response}")
                break
            
            # Step the environment with the action
            obs, reward, terminated, info = await env.step(adapter.adapt_list([action]))
            print(f"Reward: {reward}, Terminated: {terminated}")
            
            if terminated:
                break
        
        # Evaluate the result
        result = await env.evaluate()
        print(f"Final evaluation result: {result}")
    
    # Close the environment
    await env.close()
    print("Environment closed")


if __name__ == "__main__":
    asyncio.run(main()) 