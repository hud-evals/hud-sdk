from __future__ import annotations

import datetime
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

from pydantic import BaseModel, TypeAdapter

from hud.server import make_request
from hud.settings import settings
from hud.trajectory import Trajectory

logger = logging.getLogger("hud.job")

# Type variable for the decorator
T = TypeVar("T", bound=Callable)

# Global registry to store active jobs created by decorators
_ACTIVE_JOBS = {}

class Job(BaseModel):
    """
    A job represents a collection of related trajectories.
    It holds metadata and provides methods to interact with job data.
    Instances should typically be obtained via `create_job` or `load_job`.
    """

    id: str
    name: str
    metadata: dict[str, Any] | None = None
    created_at: datetime.datetime
    status: str
    
    async def load_trajectories(self, *, api_key: str | None = None) -> list[Trajectory]:
        """
        Loads the trajectories associated with this job.

        Returns:
            List[Trajectory]: The trajectories in the job
        """
        api_key = api_key or settings.api_key
        
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/v2/jobs/{self.id}/trajectories",
            api_key=api_key,
        )
        
        return TypeAdapter(list[Trajectory]).validate_python(data)


async def create_job(name: str, gym_id: str | None = None,
                     evalset_id: str | None = None,
                     metadata: dict[str, Any] | None = None) -> Job:
    """
    Creates a new job.

    Args:
        name: The name of the job
        metadata: Metadata for the job

    Returns:
        Job: The created job instance
    """
    api_key = settings.api_key
    metadata = metadata or {}

    data = await make_request(
        method="POST",
        url=f"{settings.base_url}/v2/jobs",
        json={
            "name": name,
            "metadata": metadata,
            "gym_id": gym_id,
            "evalset_id": evalset_id,
        },
        api_key=api_key,
    )
    
    # Assume the backend API returns the full job data upon creation
    # or at least the necessary fields (id, name, metadata, created_at, status)
    # If not, we might need to make a subsequent GET request
    job_data = data # Adjust if the API response structure is different
    
    return Job(
        id=job_data["id"],
        name=job_data["name"],
        metadata=job_data.get("metadata", {}), # Ensure metadata is dict
        created_at=datetime.datetime.fromisoformat(job_data["created_at"]), # Parse datetime
        status=job_data["status"],
    )


async def load_job(job_id: str, api_key: str | None = None) -> Job:
    """
    Retrieves a job by its ID.

    Args:
        job_id: The ID of the job to retrieve

    Returns:
        Job: The retrieved job instance
    """
    api_key = api_key or settings.api_key
    
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/v2/jobs/{job_id}",
        api_key=api_key,
    )
    
    if not data:
        raise ValueError(f"Job {job_id} not found")
        
    # Validate and create the Job instance from the fetched data
    return Job.model_validate(data)


def job(
    name: str,
    metadata: dict[str, Any] | None = None
) -> Callable[[T], T]:
    """
    Decorator to automatically create and associate a job with all environments
    created within the decorated function.
    
    Args:
        name: The name of the job
        metadata: Additional metadata for the job
        
    Returns:
        A decorator function that creates a job and associates it with environments
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create a job for this function call using the new function
            job = await create_job(
                name=name,
                metadata=metadata
            )
            
            # Store in global registry with a unique key based on function and call
            call_id = f"{func.__module__}.{func.__qualname__}_{id(wrapper)}"
            _ACTIVE_JOBS[call_id] = job
            
            try:
                # Add the function's frame to the stack for lookup
                frame = inspect.currentframe()
                if frame:
                    frame.f_locals["_job_call_id"] = call_id
                
                # Run the decorated function
                result = await func(*args, **kwargs)
                return result
            finally:
                # Clean up
                if call_id in _ACTIVE_JOBS:
                    del _ACTIVE_JOBS[call_id]
                    
        return cast(T, wrapper)
    return decorator


def get_active_job() -> Job | None:
    """
    Get the currently active job from the call stack, if any.
    Used internally by gym.make to automatically associate environments with jobs.
    
    Returns:
        The active job or None if no job is active
    """
    # Walk up the stack to find any frame with _job_call_id
    frame = inspect.currentframe()
    while frame:
        if "_job_call_id" in frame.f_locals:
            call_id = frame.f_locals["_job_call_id"]
            if call_id in _ACTIVE_JOBS:
                return _ACTIVE_JOBS[call_id]
        frame = frame.f_back
    
    return None
