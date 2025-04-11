#!/usr/bin/env python
"""Example demonstrating how to load Tasks from JSON and use them with environments."""

import asyncio
import json
import logging
from pathlib import Path

from hud.task import Task
from hud.gym import make

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example")

SIMPLE_TASK = """{
    "gym": "local-chrome",
    "prompt": "Open GitHub and navigate to the Human Union Data repository",
    "setup": "chrome.maximize()",
    "evaluate": "chrome.is_current_url(github.com)"
}"""

COMPLEX_TASK =  """{
    "gym": "local-chrome",
    "prompt": "Open GitHub and navigate to the Human Union Data repository",
    "setup": [
        {
            "function": "chrome.navigate",
            "args": ["https://github.com/human-union/human-union-data"]
        }
    ],
    "evaluate": [
        "chrome.is_open",
        {
            "function": "chrome.is_current_url",
            "args": ["github.com"]
        }
    ],
    "metadata": {
        "author": "example",
        "version": "1.0"
    }
}"""

async def load_and_run_task(task_json: str):
    """Load a task from JSON and run it in the environment."""
    
    # Parse the task from JSON
    task = Task.model_validate_json(task_json)
    logger.info("Loaded task: %s", task)
    logger.info("Task setup config: %s", task.setup)
    logger.info("Task evaluate config: %s", task.evaluate)
    
    # Create environment from task
    logger.info("Creating environment from task...")
    env = await make(task)
    
    try:
        # Run the evaluation (using the preloaded config)
        logger.info("Running evaluation...")
        eval_result = await env.evaluate()
        logger.info("Evaluation result: %s", eval_result)
        
    finally:
        # Make sure to close the environment
        logger.info("Closing environment...")
        await env.close()
        logger.info("Environment closed")

async def load_task_from_file(file_path: str) -> Task:
    """Load a task from a JSON file."""
    path = Path(file_path)
    with path.open() as f:
        task_json = f.read()
    return Task.model_validate_json(task_json)

async def main():
    """Run examples of loading tasks from JSON."""
    
    # Example 1: Load from JSON string
    logger.info("=== Loading simple task from JSON string ===")
    simple_task_json = SIMPLE_TASK
    simple_task = Task.model_validate_json(simple_task_json)
    logger.info("Task: %s", simple_task)
    logger.info("Task setup: %s", simple_task.setup)
    logger.info("Task evaluate: %s", simple_task.evaluate)
    
    # Example 2: Load from JSON string and run
    logger.info("\n=== Loading and running complex task ===")
    complex_task_json = COMPLEX_TASK
    await load_and_run_task(complex_task_json)
    
    # Example 3: Save and load from file (uncomment to use)
    # Save a task to a file
    # task = Task.from_json(EXAMPLE_TASKS["complex_task"])
    # with open("example_task.json", "w") as f:
    #     f.write(task.to_json())
    # logger.info("\n=== Saved task to file and loading back ===")
    # loaded_task = await load_task_from_file("example_task.json")
    # logger.info("Loaded task: %s", loaded_task)

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 