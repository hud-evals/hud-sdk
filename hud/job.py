from __future__ import annotations

import datetime
import json
from typing import Dict, List, Any, Optional

from pydantic import BaseModel

from .environment import Environment
from .server import make_request
from .settings import settings


class JobResponse(BaseModel):
    """
    Response model for job data from the API.

    Attributes:
        id: Unique identifier for the job
        name: Human-readable name of the job
        taskset: Dictionary containing taskset information
        metadata: Dictionary containing metadata
        created_at: When the job was created
        completed_at: When the job was completed, if applicable
        status: Current status of the job
    """

    id: str
    name: str
    taskset: Optional[dict[str, Any]] = None
    metadata: dict[str, Any]
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None
    status: str


async def fetch(filters: Optional[Dict[str, Any]] = None) -> List[Job]:
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
        params=params,
        api_key=api_key,
    )
    
    return [Job(**job_data) for job_data in data["jobs"]]


class Job:
    """
    A job represents a collection of related trajectories.
    
    Jobs should be created using the create class method rather than
    being constructed directly.
    """

    def __init__(
        self,
        id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime.datetime] = None,
        completed_at: Optional[datetime.datetime] = None,
        status: str = "created",
    ) -> None:
        """
        Initialize a job.

        Args:
            id: Unique identifier
            name: Human-readable name
            metadata: Optional metadata
            created_at: When the job was created
            completed_at: When the job was completed, if applicable
            status: Current status of the job
        """
        self.id = id
        self.name = name
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.datetime.now()
        self.completed_at = completed_at
        self.status = status
        self.environments: list[Environment] = []
    
    @classmethod
    async def create(cls, name: str, metadata: Optional[Dict[str, Any]] = None) -> Job:
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
            url=f"{settings.base_url}/jobs",
            json={
                "name": name,
                "metadata": json.dumps(metadata),
            },
            api_key=api_key,
        )
        
        return cls(
            id=data["id"],
            name=name,
            metadata=metadata,
            created_at=datetime.datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
            status=data.get("status", "created"),
        )
    
    @classmethod
    async def get(cls, job_id: str) -> Job:
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
            url=f"{settings.base_url}/jobs/{job_id}",
            api_key=api_key,
        )
        
        if not data:
            raise ValueError(f"Job {job_id} not found")
            
        job_data = JobResponse(**data)
        
        return cls(
            id=job_data.id,
            name=job_data.name,
            metadata=job_data.metadata,
            created_at=job_data.created_at,
            completed_at=job_data.completed_at,
            status=job_data.status,
        )
    
    async def load_trajectories(self) -> List[Any]:  # Replace Any with Trajectory once available
        """
        Loads the trajectories associated with this job.

        Returns:
            List[Trajectory]: The trajectories in the job
        """
        api_key = settings.api_key
        
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/jobs/{self.id}/trajectories",
            api_key=api_key,
        )
        
        # This is just a placeholder until Trajectory class is implemented
        return data.get("trajectories", [])