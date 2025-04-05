from pathlib import Path
from typing import Union

from ipykernel import control
from hud.env import DockerEnvController, RemoteEnvController
from hud.task import EnvSpec, PrivateEnvSpec, Task
from hud.environment.base import Environment


async def make(env_src: Union[str, EnvSpec, Task]) -> Environment:
    """
    Create an environment from an environment ID or a Task object.
    
    Args:
        env: Environment ID or Task object
    """
    env_spec = None
    if isinstance(env_src, str):
        env_spec = PrivateEnvSpec(id=env_src)
    elif isinstance(env_src, EnvSpec):
        env_spec = env_src
    elif isinstance(env_src, Task):
        env_spec = env_src.envspec

    if env_spec is None:
        raise ValueError("Could not determine environment specification from input")    
    
    # we don't yet support private env specs
    if isinstance(env_spec, PrivateEnvSpec):
        raise NotImplementedError("Private environments are not yet supported")
    
    # Create the environment (depending on location)
    if env_spec.location == "local":
        env = await DockerEnvController.create(env_spec.dockerfile)
    else:
        env = await RemoteEnvController.create(env_spec.dockerfile)
        
    # Set up the environment with either a source or env_id
    if isinstance(env_spec.controller, Path):
        env.set_source_path(env_spec.controller)
    else:
        env.set_env_id(env_spec.controller)
        
    # Create the environment itself