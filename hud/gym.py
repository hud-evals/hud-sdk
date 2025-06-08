from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.env.environment import Environment
from hud.env.local_docker_client import LocalDockerClient
from hud.env.remote_client import RemoteClient
from hud.env.remote_docker_client import RemoteDockerClient
from hud.exceptions import GymMakeException
from hud.task import Task
from hud.telemetry.context import get_current_task_run_id
from hud.types import CustomGym, Gym
from hud.utils.common import get_gym_id

if TYPE_CHECKING:
    from hud.job import Job

logger = logging.getLogger("hud.gym")


async def make(
    env_src: Gym | Task,
    *,
    job: Job | None = None,
    job_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    autolog: bool | None = None,
) -> Environment:
    """
    Create an environment from an environment ID or a Task object.

    Args:
        env_src: Environment ID or Task object
        job: Job object to associate with this environment
        job_id: ID of job to associate with this environment (deprecated, use job instead)
        metadata: Additional metadata for the environment
        autolog: Whether to autolog scores and observations (default: True for remote and False for local)
    """
    if verbose:
        logging.getLogger("hud").setLevel(logging.DEBUG)
    else:
        logging.getLogger("hud").setLevel(logging.CRITICAL)

    if autolog is None:  # Default autolog values -- otherwise, use the value passed in
        if isinstance(env_src, CustomGym):
            if env_src.location == "local":
                autolog = False  # False by default for local environments
            else:
                autolog = True  # True by default for remote environments
        else:
            autolog = True  # True by default for implicitly remote environments

    task = None
    if isinstance(env_src, str | CustomGym):
        gym = env_src
    elif isinstance(env_src, Task):
        gym = env_src.gym
        task = env_src
    else:
        raise GymMakeException(f"Invalid gym source: {env_src}", {})

    effective_job_id = None
    if job is not None:
        effective_job_id = job.id
    elif job_id is not None:
        effective_job_id = job_id
    else:
        try:
            import hud.job

            active_job = hud.job.get_active_job()
            if active_job:
                effective_job_id = active_job.id
        except ImportError:
            pass

    build_data = {}
    try:
        metadata_copy = {} if metadata is None else metadata.copy()

        current_task_run_id = get_current_task_run_id()
        if current_task_run_id:
            metadata_copy["task_run_id"] = current_task_run_id
            logger.debug(
                "Passing task_run_id %s from hud.telemetry context to environment metadata.",
                current_task_run_id,
            )

        if isinstance(gym, CustomGym):
            if isinstance(gym.image_or_build_context, str):
                uri = gym.image_or_build_context
            elif isinstance(gym.image_or_build_context, Path):
                if gym.location == "local":
                    uri, build_data = await LocalDockerClient.build_image(
                        gym.image_or_build_context
                    )
                elif gym.location == "remote":
                    uri, build_data = await RemoteDockerClient.build_image(
                        gym.image_or_build_context
                    )
                else:
                    raise ValueError(f"Invalid environment location: {gym.location}")
            else:
                raise ValueError(f"Invalid image or build context: {gym.image_or_build_context}")

            if gym.location == "local":
                logger.info("Creating local environment")
                if gym.host_config:
                    logger.info("Using host config: %s", gym.host_config)
                    client = await LocalDockerClient.create(
                        image=uri,
                        host_config=gym.host_config,
                        remote_logging_for_local_docker=autolog,  # we use autolog here
                    )
                else:
                    client = await LocalDockerClient.create(
                        image=uri,
                        remote_logging_for_local_docker=autolog,  # we use autolog here
                    )

            elif gym.location == "remote":
                logger.info("Creating remote environment")

                if gym.host_config:
                    raise ValueError("host_config is not supported for remote environments")

                client = await RemoteDockerClient.create(
                    image_uri=uri,
                    job_id=effective_job_id,
                    task_id=task.id if task else None,
                    metadata=metadata_copy,
                )
            else:
                raise ValueError(f"Invalid environment location: {gym.location}")

            if isinstance(gym.image_or_build_context, Path):
                logger.info("Setting source path %s", gym.image_or_build_context)
                client.set_source_path(gym.image_or_build_context)
        elif isinstance(gym, str):
            logger.info("Creating private environment")
            true_gym_id = await get_gym_id(gym)
            client, build_data = await RemoteClient.create(
                gym_id=true_gym_id,
                job_id=effective_job_id,
                task_id=task.id if task else None,
                metadata=metadata_copy,
            )
        else:
            raise ValueError(f"Invalid gym source: {gym}")

        environment = Environment(
            client=client,
            metadata=metadata_copy,
            task=task,
            build_data=build_data,
            autolog=autolog,
        )

        if task:
            await environment._setup()
        return environment
    except Exception as e:
        build_data["exception"] = str(e)
        raise GymMakeException("Failed to create environment", build_data) from e
