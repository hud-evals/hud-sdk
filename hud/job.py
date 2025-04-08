from __future__ import annotations

import datetime
import json
from typing import Any

from pydantic import BaseModel, TypeAdapter

from hud.server import make_request
from hud.settings import settings
from hud.trajectory import Trajectory


async def query(filters: dict[str, Any] | None = None) -> list[Job]:
    """
    Lists jobs, optionally filtered by metadata.

    Args:
        filters: Metadata filters to apply

    Returns:
        List[Job]: A list of jobs matching the filters
    """
    api_key = settings.api_key
    
    params = {}
    if filters:
        params["filters"] = json.dumps(filters)
    
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/jobs",
        json=params,
        api_key=api_key,
    )
    
    return [Job(**job_data) for job_data in data["jobs"]]


class Job(BaseModel):
    """
    A job represents a collection of related trajectories.
    
    Jobs should be created using the create class method rather than
    being constructed directly.
    """

    id: str
    name: str
    metadata: dict[str, Any]
    created_at: datetime.datetime
    status: str

    @classmethod
    async def create(cls, gym_id: str, name: str, metadata: dict[str, Any] | None = None) -> Job:
        """
        Creates a new job.

        Args:
            name: The name of the job
            metadata: Metadata for the job

        Returns:
            Job: The created job
        """
        api_key = settings.api_key
        metadata = metadata or {}

        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/runs",
            json={
                "name": name,
                "gym_id": gym_id,
                "metadata": json.dumps(metadata),
            },
            api_key=api_key,
        )
        
        return cls(
            id=data["id"],
            name=name,
            metadata=metadata,
            created_at=datetime.datetime.fromisoformat(data["created_at"]),
            status=data["status"],
        )

    @classmethod
    async def load(cls, job_id: str) -> Job:
        """
        Retrieves a job by its ID.

        Args:
            job_id: The ID of the job to retrieve

        Returns:
            Job: The retrieved job
        """
        api_key = settings.api_key
        
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/runs_v2/{job_id}",
            api_key=api_key,
        )
        
        if not data:
            raise ValueError(f"Job {job_id} not found")
            
        return cls.model_validate(data)
    
    async def load_trajectories(self) -> list[Trajectory]:
        """
        Loads the trajectories associated with this job.

        Returns:
            List[Trajectory]: The trajectories in the job
        """
        api_key = settings.api_key
        
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/runs/{self.id}/trajectories",
            api_key=api_key,
        )
        
        return TypeAdapter(list[Trajectory]).validate_python(data)
