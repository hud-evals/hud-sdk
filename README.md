# HUD SDK - Human-Agent Interaction Toolkit

A Python SDK for creating, evaluating, and benchmarking agent interactions with web browsers and OS environments.

> **Alpha Release Notice**: This SDK is currently in early release status. The API is evolving and may change in future releases as we gather feedback and improve functionality.

[![PyPI version](https://img.shields.io/pypi/v/hud-python)](https://pypi.org/project/hud-python/)

[📚 Documentation](https://documentation.hud.so) | [🏠 Homepage](https://hud.so)

## API Key Setup

Before getting started, you'll need to obtain an API key:

1. Visit [app.hud.so](https://app.hud.so) to create an account and generate your API key
2. Set it in your environment:

```bash
export HUD_API_KEY=your_api_key_here
```

## Quick Start

### Installation

```bash
pip install hud-python
```

### Simple Browser Example with Claude

```python
import os
from hud import gym
from hud.task import Task
from hud.utils import stream
from hud.agent import ClaudeAgent
from anthropic import Anthropic

# Define a simple task
task = Task(
    prompt="Insert the text 'capybara' into the search bar",
    gym="hud-browser",
    setup=("goto", "google.com"),
    evaluate=("contains_text", "capybara")
)

# Create environment
env = await gym.make(task)

# Get URLs and display live view
urls = await env.get_urls()
stream(urls["live_url"])

# Initialize Claude agent
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
agent = ClaudeAgent(anthropic)

# Agent loop
obs, _, _, _ = await env.step()
for i in range(5):
    action, done = await agent.predict(obs)
    if done:
        break
    
    obs, reward, terminated, info = await env.step(action)
    if terminated:
        break

# Evaluate and close
result = await env.evaluate()
await env.close()
```

## Documentation Sections

- **Task Creation** - Design tasks for your agents using our task specification format. Tasks define goals, success criteria, and environment configurations.

- **Environment Setup** - Create and configure browser or OS environments where your agents will operate. Environments can be remote or local.

- **Agent Integration** - Connect your AI agents (including Claude, GPT models, or custom agents) to environments and observe their interactions.

- **Evaluation** - Test agent performance with built-in evaluation methods to measure success rates, efficiency, and other metrics.

- **Advanced Features** - Build custom environments, capture telemetry data, create benchmarks, and more for detailed agent analysis.

## [Examples](examples/)

We provide several example notebooks showing how to use the HUD SDK:

1. [Browser Basics](examples/browser_use.ipynb) - Simple browser interaction with live view
2. [Task Design](examples/tasks.ipynb) - Creating and customizing tasks
3. [OSWorld](examples/osworld.ipynb) - Working with OS environments
4. [Local Development](examples/local.ipynb) - Setting up local custom environments

## Documentation

For comprehensive guides, examples, and API reference, visit [our docs](https://docs.hud.so/introduction)

## License

[MIT License](LICENSE)
