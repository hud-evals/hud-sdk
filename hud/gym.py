"""HUD gym module for creating and interacting with environments."""

import asyncio
import glob
import logging
import os
from typing import Any

from hud.environment import Environment, LocalEnvironment, RemoteEnvironment
from hud.server import make_request
from hud.settings import settings
from hud.task import Task

logger = logging.getLogger("hud.gym")

# Define the directory where local environments are stored
ENVIRONMENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "environments"
)

async def list_environments() -> list[str]:
    """List all available environments.
    
    Returns:
        list[str]: List of environment IDs
    """
    # Get local environments
    local_envs = []
    for path in glob.glob(os.path.join(ENVIRONMENTS_DIR, "local-*")):
        if os.path.isdir(path):
            env_id = os.path.basename(path)
            local_envs.append(env_id)
            logger.debug("Found local environment: %s", env_id)
    
    # Get remote environments from server
    try:
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/gyms",
            api_key=settings.api_key
        )
        remote_envs = data.get("gyms", [])
        logger.debug("Found remote environments: %s", remote_envs)
    except Exception as e:
        logger.warning("Failed to get remote environments: %s", e)
        remote_envs = []
    
    # Combine and return all environments
    return local_envs + remote_envs

def is_local_environment(env_id: str) -> bool:
    """Check if an environment is local.
    
    Args:
        env_id: Environment ID to check
        
    Returns:
        bool: True if the environment is local
    """
    # Check if it starts with "local-" and the directory exists
    if not env_id.startswith("local-") and not env_id.startswith("local_"):
        return False
        
    env_path = os.path.join(ENVIRONMENTS_DIR, env_id)
    return os.path.isdir(env_path)

async def make(
    env_id_or_task: str | Task,
    timeout: float | None = None,
    **kwargs: Any,
) -> Environment:
    """
    Create the environment and wait until it is ready.

    Args:
        env_id_or_task: The environment ID to create or a Task object
        timeout: Timeout in seconds (defaults to 60 for local, 300 for remote)
        **kwargs: Additional arguments to pass to the environment

    Returns:
        Environment: A created environment
    """
    # Extract env_id and task if provided
    if isinstance(env_id_or_task, Task):
        task = env_id_or_task
        gym_id = task.gym
    else:
        task = None
        gym_id = env_id_or_task
        
    logger.info("Creating environment: %s", gym_id)
    
    # Determine if this is a local or remote environment
    is_local = is_local_environment(gym_id)
    
    # Create the appropriate environment type
    if is_local:
        # Set default timeout for local environments
        if timeout is None:
            timeout = 60  # seconds
            
        # Create local environment without container ID
        env = LocalEnvironment(gym_id=gym_id, metadata=kwargs.get("metadata", {}))
    else:
        # Set default timeout for remote environments
        if timeout is None:
            timeout = 300  # seconds
            
        # Create remote environment
        env = RemoteEnvironment(gym_id=gym_id, metadata=kwargs.get("metadata", {}))
    
    try:
        # If we have a task, preload the setup and evaluate configurations
        if task is not None:
            if task.setup is not None:
                env.preload_setup(task.setup)
            if task.evaluate is not None:
                env.preload_evaluate(task.evaluate)
                
        # Create environment with timeout
        create_timeout = 30 if not is_local else 60
        await asyncio.wait_for(env.create_environment(), timeout=create_timeout)
        
        # Wait for environment to be ready
        await asyncio.wait_for(env.wait_for_ready(), timeout=timeout)
        return env
    except asyncio.TimeoutError:
        await env.close()
        raise TimeoutError(f"Environment {gym_id} timed out during initialization")

def list_local_environments() -> list[str]:
    """Simple function to list local environments by directory check only.
    
    Returns:
        list[str]: List of local environment IDs
    """
    result = []
    
    if not os.path.exists(ENVIRONMENTS_DIR):
        logger.warning("Environments directory not found at %s", ENVIRONMENTS_DIR)
        return []
        
    for item in os.listdir(ENVIRONMENTS_DIR):
        full_path = os.path.join(ENVIRONMENTS_DIR, item)
        if os.path.isdir(full_path) and item.startswith("local-"):
            logger.debug("Found local environment: %s", item)
            result.append(item)
            
    return result