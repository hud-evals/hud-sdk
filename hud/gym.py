from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hud.env.environment import Environment
from hud.env.local_docker_client import LocalDockerClient
from hud.env.remote_client import RemoteClient
from hud.env.remote_docker_client import RemoteDockerClient
from hud.task import Task
from hud.types import CustomGym, Gym
from hud.utils.common import get_gym_id

logger = logging.getLogger("hud.gym")

async def make(
    env_src: Gym | Task,
    *,
    job_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Environment:
    """
    Create an environment from an environment ID or a Task object.
    
    Args:
        env_src: Environment ID or Task object
    """
    if metadata is None:
        metadata = {}
    gym = None
    task = None
    if isinstance(env_src, Gym):
        gym = env_src
    elif isinstance(env_src, Task):
        gym = env_src.gym
        task = env_src

    if isinstance(gym, CustomGym):
        # Create the environment (depending on location)
        if gym.dockerfile is None:
            raise ValueError("Dockerfile is required for custom environments")
        if gym.location == "local":
            logger.info("Creating local environment")
            client = await LocalDockerClient.create(gym.dockerfile)
        elif gym.location == "remote":
            logger.info("Creating remote environment")
            client = await RemoteDockerClient.create(
                dockerfile=gym.dockerfile,
                job_id=job_id,
                task_id=task.id if task else None,
                metadata=metadata,
            )
        else:
            raise ValueError(f"Invalid environment location: {gym.location}")
            
        # Set up the environment with a source path
        if gym.controller_source_dir:
            logger.info("Setting source path")
            client.set_source_path(Path(gym.controller_source_dir))
    elif isinstance(gym, str):
        logger.info("Creating private environment")
        # Note: the gym_name_or_id is a unique identifier, but it is not a true
        # gym_id for the purposes of building the environment
        # we therefore fetch the gym_id from the HUD API here
        true_gym_id = await get_gym_id(gym)

        # Create the environment
        client = await RemoteClient.create(
            gym_id=true_gym_id,
            job_id=job_id,
            task_id=task.id if task else None,
            metadata=metadata,
        )
    else:
        raise ValueError(f"Invalid gym source: {gym}")

   # Create the environment itself
    environment = Environment(client=client, metadata=metadata, task=task)
    
    if task:
        await environment._setup()

    return environment
