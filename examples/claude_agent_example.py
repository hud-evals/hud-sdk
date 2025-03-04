#!/usr/bin/env python3
"""
Example of using Claude with the HUD-SO SDK.

This example demonstrates how to:
1. Initialize the HUD client
2. Load a gym and evalset
3. Create a run
4. Create an environment
5. Run a task with the Claude agent

Note: This requires the agent module to be in your Python path.
"""

import asyncio
import os
import sys

# Add the parent directory to the path to import the agent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hud import HUDClient
from agent.claude import Claude
from agent.response_agent import ResponseAgent
from hud.adapters.claude.adapter import ClaudeAdapter
from hud.settings import settings


async def main():
    # Initialize client with API key from settings
    api_key = settings.api_key
    if api_key is None:
        print("Error: HUD_API_KEY environment variable is not set")
        return

    # Check if ANTHROPIC_API_KEY is set
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key is None:
        print("Error: ANTHROPIC_API_KEY environment variable is not set")
        return

    # Initialize client
    client = HUDClient(api_key=api_key)
    print("Initialized HUD client")

    # Load a gym and evalset
    gym = await client.load_gym(id="OSWorld-Ubuntu")
    evalset = await client.load_evalset(id="OSWorld-Ubuntu")
    print(f"Loaded gym and evalset: {gym.id}")

    # Create a run
    run = client.create_run(
        name="claude-test-run",
        gym=gym,
        evalset=evalset,
        metadata={},
    )
    
    # Get task IDs
    tasks = await run.fetch_task_ids()
    print(f"Found {len(tasks)} tasks")

    # Create environment
    env = await run.make(metadata={"agent_id": "claude-agent"})
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
        # Use the first task for this example
        task_id = tasks[0]
        
        # Reset environment to task
        obs = await env.reset(task_id, metadata={"run": "claude-agent-run"})
        print(f"Task description: {obs.text}")
        
        # Initialize agent and adapter
        agent = Claude()
        cua_adapter = ClaudeAdapter()
        human_in_the_loop = ResponseAgent()
        
        # Maximum number of steps
        max_steps = 16
        
        # Agent loop
        for i in range(max_steps):
            # Rescale screenshot
            screenshot = cua_adapter.rescale(obs.screenshot)
            
            # Get response from agent
            response = await agent.predict(screenshot, obs.text)
            print(f"Step {i+1} response received")
            
            # Process response
            done, processed = agent.process_response(dict(response))
            
            if done:
                env.final_response = str(processed)
                
                # Check if we should continue
                if human_in_the_loop.determine_response(env.final_response) == "CONTINUE":
                    obs.text = f"Yes! Please do the following: {obs.text}"
                    obs.screenshot = None
                    continue
                break
            
            # Convert to environment actions
            actions_env = cua_adapter.adapt_list([processed])
            
            # Step environment
            obs, reward, terminated, info = await env.step(actions_env)
            
            # Exit if terminated
            if terminated:
                break
        
        # Evaluate result
        result = await env.evaluate()
        print(f"Evaluation result: {result}")
    
    # Close environment
    await env.close()
    print("Environment closed")


if __name__ == "__main__":
    asyncio.run(main()) 