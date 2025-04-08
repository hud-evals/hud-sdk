from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium

from hud.env.environment import Environment
from hud.server import make_request_sync
from hud.settings import settings


class GymnasiumWrapper(gymnasium.Env):
    """
    A wrapper to use HUD environments as Gymnasium environments.
    
    By default, HUD does not provide `terminate` and `truncate` separately in `step`.
    """
    
    def __init__(self, env: Environment) -> None:
        """
        Initialize the wrapper.
        
        Args:
            env: The HUD environment to wrap.
        """
        super().__init__()
        self.env = env
        # TODO: translate CLA Action space
        self.observation_space =  gymnasium.spaces.Dict(
            {
                # 10 MB image
                "screenshot": gymnasium.spaces.Text(max_length=10 * 1024 * 1024),
                # 1 MB text
                "text": gymnasium.spaces.Text(max_length=1 * 1024 * 1024),
            }
        )
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: The seed for random number generation
            options: Additional options for reset
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Extract setup and metadata from options if provided
        setup = options.get("setup", {}) if options else {}
        metadata = options.get("metadata", {}) if options else {}
        task_id = options.get("task_id") if options else None
        
        # Reset the environment using make_request_sync
        data = make_request_sync(
            method="POST",
            url=f"{settings.base_url}/environments/{self.env.id}/reset",
            json={"task_id": task_id, "setup": setup, "metadata": metadata},
            api_key=settings.api_key,
        )
        
        # Convert observation to gymnasium format
        info = {}
        return data["observation"], info
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Translate action to the correct format
        action_list = self.env.translate_action(action) if action is not None else []
        
        # Execute step using make_request_sync
        data = make_request_sync(
            method="POST",
            url=f"{settings.base_url}/execute_step/{self.env.id}",
            json=action_list,
            api_key=settings.api_key,
        )
        
        # Convert the raw observation to the correct type
        observation = data["observation"]
        reward = data["reward"]
        terminated = data["terminated"]
        info = data["info"]
        
        return observation, reward, terminated, False, info
    
    def close(self) -> None:
        """
        Close the environment.
        """
        make_request_sync(
            method="POST",
            url=f"{settings.base_url}/close/{self.env.id}",
            api_key=settings.api_key,
        )
    
    
