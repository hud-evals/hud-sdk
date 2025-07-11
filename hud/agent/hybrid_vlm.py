"""Hybrid VLM Agent that uses vLLM for inference and local model for updates."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import httpx
import torch

from hud.agent.base import Agent
from hud.agent.vlm import VLMAgent
from hud.adapters import Adapter, VLMAdapter
from hud.rl.types import ActionSample, Batch
from hud.utils.common import Observation

logger = logging.getLogger(__name__)


class HybridVLMAgent(VLMAgent):
    """Hybrid agent that uses vLLM for fast inference but local model for updates.
    
    This gives us:
    - Fast inference via vLLM server (no GIL, continuous batching)
    - Gradient updates on local model with LoRA
    - Best of both worlds for RL training
    """
    
    def __init__(
        self,
        # vLLM server config
        vllm_url: str = "http://localhost:8000",
        vllm_timeout: float = 30.0,
        # Local model config (passed to parent VLMAgent)
        **kwargs
    ):
        """Initialize hybrid agent.
        
        Args:
            vllm_url: URL of vLLM server
            vllm_timeout: Timeout for vLLM requests
            **kwargs: Arguments passed to VLMAgent for local model
        """
        super().__init__(**kwargs)
        
        self.vllm_url = vllm_url.rstrip("/")
        self.vllm_timeout = vllm_timeout
        self._vllm_client: Optional[httpx.AsyncClient] = None
        
    async def _get_vllm_client(self) -> httpx.AsyncClient:
        """Get or create vLLM HTTP client."""
        if self._vllm_client is None or self._vllm_client.is_closed:
            self._vllm_client = httpx.AsyncClient(
                timeout=self.vllm_timeout
            )
        return self._vllm_client
    
    async def sample(self, observation: Observation, verbose: bool = False) -> ActionSample:
        """Use vLLM for fast inference instead of local model.
        
        This overrides the parent's sample() to use vLLM server,
        which is much faster for inference.
        """
        timing_info = {}
        total_start = time.time()
        
        # Format the prompt (using parent's method)
        format_start = time.time()
        prompt = self._format_prompt(observation)
        timing_info['prompt_format_ms'] = (time.time() - format_start) * 1000
        
        # Prepare vLLM request
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "logprobs": 1,  # Request log probabilities
            "echo": False,   # Don't include prompt in response
        }
        
        # Call vLLM server
        client = await self._get_vllm_client()
        api_start = time.time()
        try:
            response = await client.post(
                f"{self.vllm_url}/v1/completions",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            logger.warning("Falling back to local model inference")
            # Fall back to parent's sample method (local model)
            return await super().sample(observation, verbose)
        timing_info['vllm_api_ms'] = (time.time() - api_start) * 1000
        
        # Extract response
        choice = data["choices"][0]
        generated_text = choice["text"]
        
        # Extract log probabilities
        log_probs = []
        tokens = []
        if "logprobs" in choice and choice["logprobs"]:
            logprobs_data = choice["logprobs"]
            if "token_logprobs" in logprobs_data:
                for token, logprob in zip(
                    logprobs_data.get("tokens", []),
                    logprobs_data.get("token_logprobs", [])
                ):
                    if logprob is not None:
                        tokens.append(token)
                        log_probs.append(logprob)
        
        total_log_prob = sum(log_probs) if log_probs else None
        
        # Parse actions
        parse_start = time.time()
        done = True
        
        processed_actions = None
        if self.adapter:
            try:
                processed_actions = self.adapter.adapt_list([generated_text])
            except Exception as e:
                logger.warning(f"Failed to adapt actions: {e}")
        timing_info['parse_actions_ms'] = (time.time() - parse_start) * 1000
        
        # Total time
        timing_info['total_ms'] = (time.time() - total_start) * 1000
        
        # Get usage stats
        usage = data.get("usage", {})
        
        return ActionSample(
            text=generated_text,
            log_probs=log_probs,
            tokens=tokens,
            total_log_prob=total_log_prob,
            actions=processed_actions,
            raw_actions=[generated_text],
            done=done,
            metadata={
                "model": self.model_name,
                "temperature": self.temperature,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "timing": timing_info,
                "inference_backend": "vllm",
            }
        )
    
    # The update() method is inherited from VLMAgent parent class
    # It will use the local model for gradient updates
    
    async def aclose(self) -> None:
        """Close HTTP client."""
        if self._vllm_client and not self._vllm_client.is_closed:
            await self._vllm_client.aclose() 