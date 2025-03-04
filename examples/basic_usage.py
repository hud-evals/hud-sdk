#!/usr/bin/env python3
"""
Basic usage example for the HUD-SO SDK.

This example shows how to:
1. Initialize the HUD client
2. Load a gym and evalset
3. Create a run
4. Create an environment
5. Run a single task
"""

import asyncio

from hud import HUDClient
from hud.settings import settings

async def main():
    # Initialize client with API key
    api_key = settings.api_key
    if api_key is None:
        print("Error: HUD_API_KEY environment variable is not set")
        return

    # Initialize client
    client = HUDClient(api_key=api_key)
    print("Initialized HUD client")

    # Load a gym
    gym = await client.load_gym(id="OSWorld-Ubuntu")
    print(f"Loaded gym: {gym.id}")

    # Load an evalset
    evalset = await client.load_evalset(id="OSWorld-Ubuntu")
    print(f"Loaded evalset: {evalset.id}")

    # Create a run
    run = client.create_run(
        name="example-run",
        gym=gym,
        evalset=evalset,
        metadata={},
    )
    print(f"Created run")

    # Fetch task IDs
    tasks = await run.fetch_task_ids()
    print(f"Found {len(tasks)} tasks")

    # Create an environment
    env = await run.make(metadata={"agent_id": "example-agent"})
    print("Created environment, waiting for it to be ready...")

    # Wait for the environment to be ready
    while True:
        state = await env.get_env_state()
        print(f"Environment state: {state}")
        if state == "running" or state == "error":
            break
        await asyncio.sleep(5)

    # Only proceed if the environment is running
    if state == "running" and tasks:
        # Take the first task
        task_id = tasks[0]
        
        # Reset environment to task
        obs = await env.reset(task_id, metadata={"run": "example-run"})
        print(f"Task description: {obs.text}")
        
        # Here you would typically implement your agent's logic
        # For this example, we'll just close the environment
        print("Example complete. Closing environment...")
    
    # Close the environment
    await env.close()
    print("Environment closed")


if __name__ == "__main__":
    asyncio.run(main()) 