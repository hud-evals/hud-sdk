# HUD

A Python SDK for creating, evaluating, and benchmarking agent interactions with web browsers and OS environments.

> **Early Release Notice**: This SDK is currently in early release status. The API is evolving and may change in future releases as we gather feedback and improve functionality.

[![PyPI version](https://img.shields.io/pypi/v/hud-python)](https://pypi.org/project/hud-python/)

[📚 Documentation](https://documentation.hud.so) | [🏠 Homepage](https://hud.so)

## API Key Setup

Before getting started, you'll need to obtain an API key:

1. Visit [app.hud.so](https://app.hud.so) to create a free account and generate your API key
2. Set it in your environment or .env file:

```bash
export HUD_API_KEY=your_api_key_here
```

## Quick Start

### Installation

```bash
pip install hud-python
```

### Simple Browser Example with Claude Computer Use

> This example uses the `@job("test-run")` decorator, so the results of this run will appear under the job named "test-run" on the your [HUD Jobs page](https://app.hud.so/jobs).

Make sure your have defined your `ANTRHOPIC_API_KEY` in environment variables to run Claude.

```python
import asyncio
from hud import gym, job
from hud.task import Task
from hud.agent import ClaudeAgent

@job("test-run")
async def main():
    task = Task(
        prompt="Insert the text 'capybara' into the search bar",
        gym="hud-browser",
        setup=("goto", "google.com"),
        evaluate=("contains_text", "capybara")
    )
    
    # Create environment using the gym module
    env = await gym.make(task)
    
    # Initialize Operator agent (API key is loaded automatically)
    agent = ClaudeAgent()
    
    # Agent loop with predict and step functions
    obs, _ = await env.reset() # Gets first observation
    for i in range(5):
        actions, done = await agent.predict(obs)

        obs, reward, terminated, info = await env.step(actions)
        if done or terminated: break
    
    # Evaluate and close
    result = await env.evaluate()
    print(f"Evaluation result: {result}")
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())

```

Alternatively, run a full evaluation set via the ```run_job``` command:

```python
from hud import load_taskset, run_job, ClaudeAgent

# load
taskset = load_taskset("GAIA")

# evaluate
job = await run_job(ClaudeAgent, taskset, "test-gaia-job")

# get results OR view them in app.hud.so
print(await job.get_analytics())
```

## Documentation Sections

Explore the core concepts and features of the SDK:

*   **[Tasks and TaskSets](https://documentation.hud.so/concepts/task)**: Define goals, context, setup, and evaluation criteria for agent scenarios. This includes both interactive and **question-answering (QA)** style tasks.
*   **[Environments](https://documentation.hud.so/concepts/environment)**: Understand the browser and OS runtimes where agents interact.
*   **[Agents](https://documentation.hud.so/concepts/agent)**: Learn about the agent architecture (Claude, Operator) and how they process observations and predict actions.
*   **[Adapters](https://documentation.hud.so/concepts/adapter)**: See how actions and observations are translated between agents and environments.
*   **[Jobs](https://documentation.hud.so/concepts/job)**: Group related runs for analysis and viewing on the HUD platform.
*   **[Trajectories](https://documentation.hud.so/concepts/trajectory)**: Understand the recorded data from each agent run.
*   **Advanced Topics**:
    *   **[CLA Action Details](https://documentation.hud.so/advanced/cla-details)**: Explore the standardized action format.
    *   **[Custom Environments](https://documentation.hud.so/advanced/custom-environments)**: Build your own Docker-based local or remote environments.
    *   **[Advanced Environment Control](https://documentation.hud.so/advanced/environment-control)**: Use `invoke`, `execute`, and `_setup` for finer control.

*   **[Full API Reference](https://documentation.hud.so/api-reference/gym)**: Detailed specifications for all modules and classes.

## [Examples](examples/)

We recommend you first take a look at the example notebooks showing how to use the HUD SDK:

1. [Browser Basics](examples/browser_use.ipynb) - Simple browser interaction with live view
2. [Task Design](examples/tasks.ipynb) - Creating and customizing tasks
3. [OSWorld](examples/osworld.ipynb) - Running the OSWorld benchmark
4. [Local Development](examples/local.ipynb) - Setting up local custom environments

## Documentation

For comprehensive guides, examples, and API reference, visit [our docs](https://documentation.hud.so/introduction)

## License

[MIT License](LICENSE)

## Citation

If you use this SDK in your research, please cite it as follows:

```bibtex
@software{hud2025agentevalplatform,
  author = {HUD and Jay Ram and Lorenss Martinsons and Parth Patel and Oskars Putans and Govind Pimpale and Mayank Singamreddy and Nguyen Nhat Minh},
  title = {{HUD: An Evaluation Platform for Agents}},
  date = {2025-04},
  url = {https://github.com/hud-evals/hud-sdk},
  langid = {en}
}
```