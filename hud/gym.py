from __future__ import annotations

import logging
from typing import Any

from hud.env.docker_client import DockerClient
from hud.env.environment import Environment
from hud.env.remote_client import RemoteClient
from hud.server.requests import make_request
from hud.settings import settings
from hud.task import Task
from hud.types import CustomGym, Gym

logger = logging.getLogger("hud.gym")



async def _get_gym_id(gym_name_or_id: str) -> str:
    """
    Get the gym ID for a given gym name or ID.
    """
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/v1/gyms/{gym_name_or_id}",
        api_key=settings.api_key,
    )

    return data["id"]


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
        if gym.location == "local":
            logger.info("Creating local environment")
            client = await DockerClient.create(gym.dockerfile)
        elif gym.location == "remote":
            logger.info("Creating remote environment")
            
            true_gym_id = await _get_gym_id("local-docker")
            
            # augment metadata with dockerfile
            if "environment_config" not in metadata:
                metadata["environment_config"] = {}
            metadata["environment_config"]["dockerfile"] = gym.dockerfile

            client = await RemoteClient.create(
                gym_id=true_gym_id,
                job_id=job_id,
                task_id=task.id if task else None,
                metadata=metadata,
            )
        else:
            raise ValueError(f"Invalid environment location: {gym.location}")
            
        # Set up the environment with a source path
        if gym.controller_source_dir:
            logger.info("Setting source path")
            client.set_source_path(gym.controller_source_dir)
    elif isinstance(gym, str):
        logger.info("Creating private environment")
        # Note: the gym_name_or_id is a unique identifier, but it is not a true
        # gym_id for the purposes of building the environment
        # we therefore fetch the gym_id from the HUD API here
        true_gym_id = await _get_gym_id(gym)

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
