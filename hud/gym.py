from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from hud.env.docker_env_client import DockerEnvClient
from hud.env.environment import Environment
from hud.env.remote_env_client import RemoteEnvClient
from hud.job import Job
from hud.server.requests import make_request
from hud.settings import settings
from hud.task import Task
from hud.types import EnvSpec, PrivateEnvSpec, PublicEnvSpec

logger = logging.getLogger("hud.gym")

def _default_job_name() -> str:
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return f"Untitled {current_time}"


async def _get_gym_id(gym_name_or_id: str) -> str:
    """
    Get the gym ID for a given gym name or ID.
    """
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/gyms/{gym_name_or_id}",
        api_key=settings.api_key,
    )

    return data["id"]


async def make(
    env_src: str | EnvSpec | Task,
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
    env_spec = None
    setup = None
    evaluate = None
    task = None
    if isinstance(env_src, str):
        env_spec = PrivateEnvSpec(gym_id=env_src)
    elif isinstance(env_src, EnvSpec):
        env_spec = env_src
    elif isinstance(env_src, Task):
        env_spec = env_src.envspec
        setup = env_src.setup
        evaluate = env_src.evaluate
        task = env_src

    if isinstance(env_spec, PrivateEnvSpec):
        logger.info("Creating private environment")


        # Note: the gym_id is a unique identifier, but it is not a true
        # gym_id for the purposes of building the environment
        # we therefore fetch the gym_id from the HUD API here
        true_gym_id = await _get_gym_id(env_spec.gym_id)

        # Create a job if one is not provided
        if job_id is None:
            job_name = _default_job_name()
            logger.info("No job ID provided, creating a new job %s", job_name)
            job = await Job.create(gym_id=true_gym_id, name=job_name)
            job_id = job.id

        # Create the environment
        client = await RemoteEnvClient.create(gym_id=true_gym_id, job_id=job_id, metadata=metadata)
    elif isinstance(env_spec, PublicEnvSpec):
        # Create the environment (depending on location)
        if env_spec.location == "local":
            logger.info("Creating local environment")
            client = await DockerEnvClient.create(env_spec.dockerfile)
        elif env_spec.location == "remote":
            logger.info("Creating remote environment")
            raise NotImplementedError(
                "Remote dockerfile environments are not yet supported"
            )
            client = await RemoteEnvClient.create(
                dockerfile=env_spec.dockerfile,
                metadata=metadata,
            )
        else:
            raise ValueError(f"Invalid environment location: {env_spec.location}")
            
        # Set up the environment with a source path
        if env_spec.controller_source_dir:
            logger.info("Setting source path")
            client.set_source_path(env_spec.controller_source_dir)
    else:
        raise ValueError(f"Invalid environment source: {env_src}")

   # Create the environment itself
    environment =  Environment(client=client, metadata=metadata)
    
    if task:
        environment.task = task

    if setup:
        logger.info("Preloading setup")
        environment.preload_setup(setup)
    if evaluate:
        logger.info("Preloading evaluate")
        environment.preload_evaluate(evaluate)


    return environment
