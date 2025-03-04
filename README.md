# HUD-SO: Human Union Data SDK (Alpha Release)

A Python SDK for interacting with HUD environments and evaluation benchmarks.

> **Note**: This is an alpha release (v0.1.0-alpha). APIs may change significantly in future releases.

[![PyPI version](https://badge.fury.io/py/hud-so.svg)](https://badge.fury.io/py/hud-so)
[![CI](https://github.com/humanuniondata/hud-so/actions/workflows/ci.yml/badge.svg)](https://github.com/humanuniondata/hud-so/actions/workflows/ci.yml)

## Installation

```bash
# Install the latest stable release
pip install hud-so

# Install the latest alpha release (may include breaking changes)
pip install --pre hud-so

# Install a specific alpha version
pip install hud-so==0.1.0-alpha
```

## Quick Start

```python
import asyncio
from hud import HUDClient
from hud.settings import settings

async def main():
    # Initialize client with API key from settings
    client = HUDClient(api_key=settings.api_key)
    
    # Load gym and evalset
    gym = await client.load_gym(id="OSWorld-Ubuntu")
    evalset = await client.load_evalset(id="OSWorld-Ubuntu")
    
    # Create a run and get tasks
    run = client.create_run(name="example-run", gym=gym, evalset=evalset)
    tasks = await run.fetch_task_ids()
    
    # Create environment and wait for it to be ready
    env = await run.make(metadata={"agent_id": "example"})
    while True:
        if await env.get_env_state() in ["running", "error"]:
            break
        await asyncio.sleep(2)
    
    # Run a task
    if tasks:
        obs = await env.reset(tasks[0], metadata={"run": "example"})
        print(f"Task: {obs.text}")
    
    # Close when done
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Custom Agent Loop

Here's how to create a custom agent that interfaces with the HUD environment:

```python
import asyncio
from typing import Any, Dict, Tuple

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
    client = HUDClient(api_key=settings.api_key)
    
    # Load gym and evalset
    gym = await client.load_gym(id="OSWorld-Ubuntu")
    evalset = await client.load_evalset(id="OSWorld-Ubuntu")
    
    # Create the run
    run = client.create_run(name="simple-agent-run", gym=gym, evalset=evalset)
    tasks = await run.fetch_task_ids()
    
    # Initialize the agent and adapter
    agent = SimpleAgent()
    adapter = SimpleAdapter()
    
    # Create environment with the adapter
    env = await run.make(adapter=adapter, metadata={"agent_id": "simple-agent"})
    
    # Wait for environment to be ready
    while True:
        if await env.get_env_state() in ["running", "error"]:
            break
        await asyncio.sleep(2)
    
    # Reset to a task
    if tasks:
        obs = await env.reset(tasks[0], metadata={"run": "simple-agent-run"})
        
        # Agent interaction loop
        max_steps = 5
        for i in range(max_steps):
            # Get agent's prediction
            response = await agent.predict(obs.screenshot, obs.text)
            
            # Process the response
            done, action = agent.process_response(response)
            
            if done:
                # This is a final response
                env.final_response = str(action)
                break
            
            # Step the environment with the action
            obs, reward, terminated, info = await env.step(adapter.adapt_list([action]))
            
            if terminated:
                break
        
        # Evaluate the result
        result = await env.evaluate()
        print(f"Evaluation result: {result}")
    
    # Close the environment
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Features

- Connect to HUD evaluation environments
- Run benchmarks across various tasks
- Support for different agent adapters
- Asynchronous API for efficient interaction

## Environment Variables

The SDK uses environment variables for configuration. You can set these in your environment or in a `.env` file in your project root:

```
# .env file
HUD_API_KEY=your-api-key-here
```

Required:
- `HUD_API_KEY`: Your HUD API key (required)

## Examples

See the `examples/` directory for more detailed examples of:
- Basic API usage
- Custom agent implementation
- Claude agent integration

## Development

```bash
# Create a virtual environment (option 1: venv)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Create a virtual environment (option 2: uv)
uv venv
# Activate according to your shell (e.g., .venv\Scripts\activate on Windows)

# Install in development mode with pip
pip install -e ".[dev]"

# Or with uv (recommended)
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint code
ruff check .
```

## License

[MIT License](LICENSE)
