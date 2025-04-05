from pathlib import Path
from typing import Union, Any
import logging

from hud.env.docker_env_client import DockerEnvClient
from hud.env.remote_env_client import RemoteEnvClient
from hud.task import Task
from hud.types import PrivateEnvSpec, EnvSpec
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
        env_spec = PrivateEnvSpec(id=env_src)
    elif isinstance(env_src, EnvSpec):
        env_spec = env_src
    elif isinstance(env_src, Task):
        env_spec = env_src.envspec
        setup = env_src.setup
        evaluate = env_src.evaluate

    if env_spec is None:
        raise ValueError("Could not determine environment specification from input")    
    
    # we don't yet support private env specs
    if isinstance(env_spec, PrivateEnvSpec):
        raise NotImplementedError("Private environments are not yet supported")
    
    # Create the environment (depending on location)
    if env_spec.location == "local":
        logger.info("Creating local environment")
        client = await DockerEnvClient.create(env_spec.dockerfile)
    else:
        logger.info("Creating remote environment")
        client = await RemoteEnvClient.create(env_spec.dockerfile)
        
    # Set up the environment with either a source or env_id
    if isinstance(env_spec.controller, Path):
        logger.info("Setting source path")
        client.set_source_path(env_spec.controller)
    else:
        logger.info("Setting env id")
        client.set_env_id(env_spec.controller)
        
    # Create the environment itself
    environment =  Environment(client=client, metadata=metadata)
    
    if setup:
        logger.info("Preloading setup")
        environment.preload_setup(setup)
    if evaluate:
        logger.info("Preloading evaluate")
        environment.preload_evaluate(evaluate)

    return environment