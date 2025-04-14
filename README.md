# HUD

A Python SDK for interacting with HUD environments and evaluation benchmarks for browser use and computer use models.

> **Alpha Release Notice**: This SDK is currently in early release status. The API is evolving and may change in future releases as we gather feedback and improve functionality.

[![PyPI version](https://img.shields.io/pypi/v/hud-python)](https://pypi.org/project/hud-python/)

[📚 Documentation](https://documentation.hud.so) | [🏠 Homepage](https://hud.so)


## Quick start

[RECOMMENDED] To set get started with an agent, see the [Claude Computer use example](https://github.com/Human-Data/hud-sdk/tree/main/examples).

Install the package with Python>=3.9:
```bash
pip install hud-python
```

Make sure to setup your account with us (email founders@hud.so) and add your API key to the environment variables:
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

## Documentation

For comprehensive guides, examples, and API reference, visit [our docs](https://docs.hud.so/introduction)

## License

[MIT License](LICENSE)

## Citation

If you use this SDK in your research, please cite it as follows:

## Citation

If you use this SDK in your research, please cite it as follows:

## Citation

If you use this SDK in your research, please cite it as follows:

```bibtex
@software{hud2025agentevalplatform,
  author = {HUD and Jay Ram and Lorenss Martinsons and Parth Patel and Oskars Putans and Govind Pimpale and Mayank Singamreddy and Nguyen Nhat Minh},
  title = {{HUD: An Evaluation Platform for Computer Use Agents}},
  date = {2025-03},
  url = {https://github.com/Human-Data/hud-sdk},
  langid = {en}
}
```