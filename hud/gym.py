from pathlib import Path
from typing import Union, Any
import logging

from hud.env.docker_env_client import DockerEnvClient
from hud.env.remote_env_client import RemoteEnvClient
from hud.task import Task
from hud.types import PrivateEnvSpec, EnvSpec, PublicEnvSpec
from hud.env.environment import Environment

logger = logging.getLogger("hud.gym")

async def make(env_src: Union[str, EnvSpec, Task], *, metadata: dict[str, Any] = {}) -> Environment:
    """
    Create an environment from an environment ID or a Task object.
    
    Args:
        env_src: Environment ID or Task object
    """
    env_spec = None
    setup = None
    evaluate = None
    if isinstance(env_src, str):
        env_spec = PrivateEnvSpec(gym_id=env_src)
    elif isinstance(env_src, EnvSpec):
        env_spec = env_src
    elif isinstance(env_src, Task):
        env_spec = env_src.envspec
        setup = env_src.setup
        evaluate = env_src.evaluate
    
    if isinstance(env_spec, PrivateEnvSpec):
        logger.info("Creating private environment")
        client = await RemoteEnvClient.create(gym_id=env_spec.gym_id)        
    elif isinstance(env_spec, PublicEnvSpec):
        # Create the environment (depending on location)
        if env_spec.location == "local":
            logger.info("Creating local environment")
            client = await DockerEnvClient.create(env_spec.dockerfile)
        elif env_spec.location == "remote":
            logger.info("Creating remote environment")
            raise NotImplementedError("Remote dockerfile environments are not yet supported")
            client = await RemoteEnvClient.create_from_dockerfile(env_spec.dockerfile)
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
    
    if setup:
        logger.info("Preloading setup")
        environment.preload_setup(setup)
    if evaluate:
        logger.info("Preloading evaluate")
        environment.preload_evaluate(evaluate)

    return environment