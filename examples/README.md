# HUD SDK Examples

This directory contains example notebooks demonstrating different aspects of the HUD SDK.

## Getting Started

1. **Installation**: Follow the [main README](../README.md) instructions to install the SDK.

2. **API Key**: Get your API key from [app.hud.so](https://app.hud.so) and set it as an environment variable:
   ```bash
   export HUD_API_KEY=your_api_key_here
   ```

3. **Starting with Examples**:
   - [browser_use.ipynb](browser_use.ipynb) - Begin here for browser-based agent interaction with a live view
   - [tasks.ipynb](tasks.ipynb) - Learn how to create and customize tasks for different environments
   - [osworld.ipynb](osworld.ipynb) - Explore OS-based environments with Claude agent integration
   - [local.ipynb](local.ipynb) - Develop and test with local custom environments

## Key Concepts

- **Tasks**: Define the objective, context, and success criteria for what an agent should accomplish
- **Environments**: Browser or OS interfaces where agents can perceive and interact with real applications
- **Agents**: AI systems (like Claude or OpenAI models) that process observations and generate actions in environments
- **Jobs**: Group related environment runs (trajectories) together for evaluation and analysis
- **Evaluation**: Methods to assess agent performance, success rates, and behavior patterns

Each example demonstrates practical applications of these concepts with code you can run and modify.

For more detailed documentation, visit [docs.hud.so](https://docs.hud.so/introduction).


