"""OpenAI-compatible agent for vLLM and other OpenAI API servers."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
import json

from hud.agent.base import Agent
from hud.adapters import Adapter, VLMAdapter
from hud.rl.types import ActionSample
from hud.utils.common import Observation

logger = logging.getLogger(__name__)


class OpenAICompatibleAgent(Agent[httpx.AsyncClient, Dict[str, Any]]):
    """Agent that works with OpenAI-compatible APIs like vLLM.
    
    This agent:
    - Supports any OpenAI-compatible API (vLLM, TGI, etc.)
    - Implements sample() with log probabilities for RL training
    - Uses the completions API with logprobs enabled
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str = "EMPTY",  # vLLM doesn't require API key
        adapter: Optional[Adapter] = None,
        name: Optional[str] = None,
        # Generation configuration
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        # System prompt
        system_prompt: Optional[str] = None,
        # Performance settings
        timeout: float = 60.0,
    ):
        """Initialize OpenAI-compatible agent.
        
        Args:
            base_url: Base URL for the API (e.g., "http://localhost:8000/v1")
            model_name: Model name to use
            api_key: API key (use "EMPTY" for vLLM)
            adapter: Adapter for parsing text to actions
            name: Agent name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            system_prompt: System prompt for the model
            timeout: Request timeout in seconds
        """
        super().__init__(client=None, adapter=adapter, name=name)
        
        # Use VLMAdapter as default if no adapter provided
        if self.adapter is None:
            self.adapter = VLMAdapter()
        
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        
    def _default_system_prompt(self) -> str:
        """Default system prompt for UI automation."""
        return (
            "You are an AI assistant that helps users interact with computer interfaces. "
            "You can see screenshots and perform actions like clicking, typing, and scrolling.\n\n"
            "Respond with your thoughts and actions in this format:\n"
            "Thought: [Your reasoning about what you see and what to do]\n"
            "Action: [action_type] [parameters]\n\n"
            "Available actions:\n"
            "- click x,y (click at coordinates)\n"
            "- type \"text\" (type the given text)\n"
            "- scroll up/down (scroll in direction)\n"
            "- done \"response\" (task complete with final response)\n"
        )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout
            )
        return self._client
    
    def _format_prompt(self, observation: Observation) -> str:
        """Format observation into a prompt."""
        messages = []
        
        # System message
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # User message
        user_content = []
        if observation.text:
            user_content.append(observation.text)
        
        if observation.screenshot:
            # For now, just indicate presence
            # vLLM doesn't support images yet, but TGI does
            user_content.append("[Screenshot provided]")
        
        messages.append({"role": "user", "content": "\n".join(user_content)})
        
        # Convert to prompt format
        # This is a simple format, you might need to adjust based on the model
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def fetch_response(self, observation: Observation) -> Tuple[List[Dict[str, Any]], bool]:
        """Fetch response (for compatibility with base class)."""
        sample = await self.sample(observation)
        return [{"text": sample.text}], sample.done
    
    async def sample(self, observation: Observation, verbose: bool = False) -> ActionSample:
        """Generate text with log probabilities using OpenAI-compatible API.
        
        Returns ActionSample with:
        - Generated text
        - Token log probabilities
        - Parsed actions
        - Task completion status
        """
        timing_info = {}
        total_start = time.time()
        
        client = await self._get_client()
        
        # Format the prompt
        format_start = time.time()
        prompt = self._format_prompt(observation)
        timing_info['prompt_format_ms'] = (time.time() - format_start) * 1000
        
        # Prepare request
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": 1,  # Request log probabilities
            "echo": False,   # Don't include prompt in response
        }
        
        # Make request
        api_start = time.time()
        try:
            response = await client.post("/completions", json=request_data)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise RuntimeError(f"OpenAI API request failed: {e}")
        timing_info['api_call_ms'] = (time.time() - api_start) * 1000
        
        # Extract response
        choice = data["choices"][0]
        generated_text = choice["text"]
        
        # Extract log probabilities
        log_probs = []
        tokens = []
        if "logprobs" in choice and choice["logprobs"]:
            logprobs_data = choice["logprobs"]
            if "token_logprobs" in logprobs_data:
                # Skip None values (first token)
                for i, (token, logprob) in enumerate(zip(
                    logprobs_data.get("tokens", []),
                    logprobs_data.get("token_logprobs", [])
                )):
                    if logprob is not None:
                        tokens.append(token)
                        log_probs.append(logprob)
        
        total_log_prob = sum(log_probs) if log_probs else None
        
        # Parse actions
        parse_start = time.time()
        done = True  # Simplified for now
        
        processed_actions = None
        if self.adapter:
            try:
                processed_actions = self.adapter.adapt_list([generated_text])
            except Exception as e:
                logger.warning(f"Failed to adapt actions: {e}")
        timing_info['parse_actions_ms'] = (time.time() - parse_start) * 1000
        
        # Total time
        timing_info['total_ms'] = (time.time() - total_start) * 1000
        
        # Get usage stats if available
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
            }
        )
    
    async def aclose(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose() 