"""Hosted VLM agent that calls remote endpoints for sampling and updates."""

from __future__ import annotations

import httpx
import logging
from typing import List, Optional, Dict, Any
import asyncio

from pydantic import TypeAdapter

from hud.adapters.common.types import CLA
from hud.agent.base import Agent
from hud.adapters import Adapter
from hud.rl.types import ActionSample, Batch
from hud.utils.common import Observation

logger = logging.getLogger(__name__)


class HostedVLMAgent(Agent[httpx.AsyncClient, str]):
    """Agent that talks to a deployed VLMAgent via HTTP endpoints.
    
    This is a thin wrapper that:
    - Calls /sample for inference with log probabilities
    - Calls /update for gradient updates
    - Otherwise identical interface to VLMAgent
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        adapter: Optional[Adapter] = None,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """Initialize hosted VLM agent.
        
        Args:
            base_url: Base URL of the deployed agent
            adapter: Optional adapter for action processing
            name: Agent name
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        super().__init__(client=None, adapter=adapter, name=name)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create HTTP client."""
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    headers = {}
                    if self._api_key:
                        headers["Authorization"] = f"Bearer {self._api_key}"
                    self._client = httpx.AsyncClient(
                        headers=headers,
                        timeout=self._timeout
                    )
        return self._client

    async def fetch_response(self, observation: Observation) -> tuple[List[str], bool]:
        """Fetch response (for compatibility with base class)."""
        # Use sample() for full RL support
        sample = await self.sample(observation)
        return [sample.text], sample.done

    async def sample(self, observation: Observation, verbose: bool = False) -> ActionSample:
        """Sample action by calling remote /sample endpoint.
        
        Returns ActionSample with:
        - Generated text
        - Log probabilities (if server provides them)
        - Parsed actions
        - Task completion status
        """
        client = await self._get_client()
        
        # Prepare request payload
        payload = {
            "observation": {
                "text": observation.text,
                "screenshot": observation.screenshot
            },
            "verbose": verbose
        }
        
        try:
            resp = await client.post(
                f"{self._base_url}/sample",
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to call /sample endpoint: {e}")
            raise RuntimeError(f"Sample request failed: {e}")
        
        # Deserialize ActionSample from response
        action_sample_data = data.get("action_sample", {})
        
        # Reconstruct CLA actions from server response
        # Server already did the adaptation, we just need to reconstruct CLA objects
        actions_data = action_sample_data.get("actions", [])
        cla_actions = []

        if self.adapter is not None:
            cla_actions = self.adapter.adapt_list(actions_data)
        else:
            cla_actions = [TypeAdapter(CLA).validate_python(action_dict) for action_dict in actions_data]
        
        return ActionSample(
            text=action_sample_data.get("text", ""),
            log_probs=action_sample_data.get("log_probs"),
            tokens=action_sample_data.get("tokens"),
            total_log_prob=action_sample_data.get("total_log_prob"),
            actions=cla_actions,
            raw_actions=action_sample_data.get("raw_actions", []),
            done=action_sample_data.get("done", False),
            metadata=action_sample_data.get("metadata", {})
        )
    
    async def update(self, batch: Batch) -> Dict[str, float]:
        """Perform gradient update by calling remote /update endpoint.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of training statistics
        """
        client = await self._get_client()
        
        # Serialize batch for transport
        # Convert observations to simple dicts
        observations = []
        for obs in batch.observations:
            observations.append({
                "text": obs.text,
                "screenshot": obs.screenshot
            })
        
        payload = {
            "batch": {
                "observations": observations,
                "texts": batch.texts,
                "advantages": batch.advantages,
                "returns": batch.returns,
                "old_log_probs": batch.old_log_probs,
                "metadata": batch.metadata
            }
        }
        
        try:
            resp = await client.post(
                f"{self._base_url}/update",
                json=payload,
                timeout=self._timeout * 2  # Updates may take longer
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to call /update endpoint: {e}")
            raise RuntimeError(f"Update request failed: {e}")
        
        # Extract stats from response
        stats = data.get("stats", {})
        
        # Ensure all values are floats
        float_stats = {}
        for k, v in stats.items():
            try:
                float_stats[k] = float(v)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert stat {k}={v} to float, skipping")
        
        return float_stats
    
    async def aclose(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose() 