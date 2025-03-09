"""
HUD client for interacting with the API.
"""

from __future__ import annotations

import json
from typing import Any

from .adapters.common import Adapter
from .environment import EvalSet
from .gym import Gym
from .run import Run, RunResponse
from .server import make_request
from .settings import settings


class HUDClient:
    """
    Client for interacting with the HUD API.

    This is the main entry point for the SDK, providing methods to load gyms,
    evalsets, and create runs.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the HUD client with an API key.

        Args:
            api_key: API key for authentication with the HUD API
        """
        self.api_key = api_key or settings.api_key
        settings.api_key = self.api_key

    async def load_gym(self, id: str) -> Gym:
        """
        Load a gym by ID from the HUD API.

        Args:
            id: The ID of the gym to load

        Returns:
            Gym: The loaded gym object
        """
        # API call to get gym info
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/gyms/{id}",
            api_key=self.api_key,
        )
        return Gym(id=data["id"], name=data["name"])

    async def load_evalset(self, id: str) -> EvalSet:
        """
        Load an evalset by ID from the HUD API.

        Args:
            id: The ID of the evalset to load

        Returns:
            EvalSet: The loaded evalset object
        """
        # API call to get evalset info
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/evalsets/{id}",
            api_key=self.api_key,
        )
        return EvalSet(id=data["id"], name=data["name"])

    async def list_gyms(self) -> list[str]:
        """
        List all available gyms.

        Returns:
            list[str]: List of gym IDs
        """
        # API call to get gyms
        data = await make_request(
            method="GET", url=f"{settings.base_url}/gyms", api_key=self.api_key
        )
        return data["gyms"]

    async def get_runs(self) -> list[Run]:
        """
        Get all runs associated with the API key.

        Returns:
            list[Run]: List of run objects
        """
        # API call to get runs
        data = await make_request(
            method="GET", url=f"{settings.base_url}/runs", api_key=self.api_key
        )
        return data["runs"]

    async def load_run(self, id: str, adapter: Adapter | None = None) -> Run | None:
        """
        Load a run by ID from the HUD API.

        Args:
            id: The ID of the run to load
            adapter: Optional adapter for action conversion

        Returns:
            Run: The loaded run object, or None if not found
        """
        adapter = adapter or Adapter()
        # API call to get run info
        data = await make_request(
            method="GET",
            url=f"{settings.base_url}/runs/{id}",
            api_key=self.api_key,
        )
        if data:
            response = RunResponse(**data)
            gym = Gym(id=response.gym["id"], name=response.gym["name"])
            evalset = EvalSet(
                id=response.evalset["id"],
                name=response.evalset["name"],
                tasks=response.evalset["tasks"],
            )
            return Run(
                id=response.id,
                name=response.name,
                gym=gym,
                evalset=evalset,
                adapter=adapter,
                config=response.config,
                metadata=response.metadata,
            )
        return None

    async def create_run(
        self,
        name: str,
        gym: Gym,
        evalset: EvalSet,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        adapter: Adapter | None = None,
    ) -> Run:
        """
        Create a new run in the HUD system.

        Args:
            name: Name of the run
            gym: Gym to use for the run
            evalset: Evalset to use for the run
            config: Optional configuration parameters
            metadata: Optional metadata for the run
            adapter: Optional adapter for action conversion

        Returns:
            Run: The created run object
        """
        adapter = adapter or Adapter()
        # Make synchronous API call to create run
        if metadata is None:
            metadata = {}
        if config is None:
            config = {}
        data = await make_request(
            method="POST",
            url=f"{settings.base_url}/runs",
            json={
                "name": name,
                "gym_id": gym.id,
                "evalset_id": evalset.id,
                "config": json.dumps(config),
                "metadata": json.dumps(metadata),
            },
            api_key=self.api_key,
        )
        return Run(
            id=data["id"],
            name=name,
            gym=gym,
            evalset=evalset,
            adapter=adapter,
            config=config,
            metadata=metadata,
        )

    def display_stream(self, live_url: str) -> None:
        """
        Display a stream in the HUD system.
        """
        from IPython.display import HTML, display

        html_content = f"""
        <div style="width: 960px; height: 540px; overflow: hidden;">
            <div style="transform: scale(0.5); transform-origin: top left;">
                <iframe src="{live_url}" width="1920" height="1080" style="border: 1px solid #ddd;">
                </iframe>
            </div>
        </div>
        """
        display(HTML(html_content))
