# HUD SDK (Alpha Release)

A Python SDK for interacting with HUD environments and evaluation benchmarks for browser use and computer use models.

Visit [hud.so](https://hud.so) for more information about HUD.

> **Alpha Release Notice**: This SDK is currently in alpha status (v0.1.0-alpha). The API is still evolving and may change in future releases as we gather feedback and improve functionality.

[![PyPI version](https://img.shields.io/pypi/v/hud-python)](https://pypi.org/project/hud-python/)

[📚 Documentation](https://documentation.hud.so) | [🏠 Homepage](https://hud.so)

## Quick Start

```bash
# Install the latest stable release
pip install hud-python

# Install the latest alpha release (may include breaking changes)
pip install --pre hud-python

# Install a specific alpha version
pip install hud-python==0.1.0-alpha
```

```python
import asyncio
from hud import HUDClient

async def main():
    # Initialize client with API key
    client = HUDClient(api_key="your-api-key")
    
    # Load a gym and evaluation set
    gym = await client.load_gym(id="OSWorld-Ubuntu")
    evalset = await client.load_evalset(id="OSWorld-Ubuntu")
    
    # Create a run and environment
    run = client.create_run(name="example-run", gym=gym, evalset=evalset)
    env = await run.make(metadata={"agent_id": "example"})

    # Agent loop goes here
    # For complete examples and usage guides, see our documentation

    # Close the environment when done
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Features

- Connect to HUD evaluation environments
- Run benchmarks across various tasks
- Support for different agent adapters
- Asynchronous API for efficient interaction

## Documentation

For comprehensive guides, examples, and API reference, visit:
- [Getting Started](https://docs.hud.so/introduction)
- [Installation](https://docs.hud.so/installation)
- [API Reference](https://docs.hud.so/api-reference)
- [Examples](https://docs.hud.so/examples)

## License

[MIT License](LICENSE)
