# HUD

A Python SDK for interacting with HUD environments and evaluation benchmarks for browser use and computer use models. Visit [hud.so](https://hud.so).

> **Alpha Release Notice**: This SDK is currently in alpha status (v0.1.0-alpha). The API is evolving and may change in future releases as we gather feedback and improve functionality.

[![PyPI version](https://img.shields.io/pypi/v/hud-python)](https://pypi.org/project/hud-python/)

[📚 Documentation](https://documentation.hud.so) | [🏠 Homepage](https://hud.so)


## Quick start

[RECOMMENDED] To set get started with an agent, see the [Claude Computer use example](https://github.com/Human-Data/hud-sdk/tree/main/examples).


Otherwise, install the package with Python>=3.9:
```bash
pip install hud-python
```

Make sure to setup your account [here](https://hud.so/settings) and add your API key to the environment variables:
```bash
HUD_API_KEY=<your-api-key>
```

Load in your agent and create a run! Go to the [examples](https://github.com/Human-Data/hud-sdk/tree/main/examples) folder for more examples.
```python
import asyncio
from hud import HUDClient

async def main():
    # Initialize client with API key
    client = HUDClient(api_key=os.getenv("HUD_API_KEY"))
    
    # Load a gym and evaluation set
    gym = await client.load_gym(id="OSWorld-Ubuntu")
    evalset = await client.load_evalset(id="OSWorld-Ubuntu")
    
    # Create a run and environment
    run = await client.create_run(name="example-run", gym=gym, evalset=evalset)
    env = await run.make(metadata={"agent_id": "OSWORLD-1"})
    await env.wait_for_ready()
    
    ### 
    ### Agent loop goes here, see example in /examples
    ###

    # Evaluate the environment
    result = await env.evaluate()

    # Close the environment when done
    await env.close()

    # Get analytics for the run such as rewards, task completions, etc.
    analytics = await run.get_analytics()
    print(analytics)

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

- Connect to HUD evaluation environments
- Run benchmarks across various tasks
- Support for different agent adapters
- Asynchronous API

## Documentation

For comprehensive guides, examples, and API reference, visit:
- [Getting Started](https://docs.hud.so/introduction)
- [Installation](https://docs.hud.so/installation)
- [API Reference](https://docs.hud.so/api-reference)
- [Examples](https://docs.hud.so/examples)

## License

[MIT License](LICENSE)
