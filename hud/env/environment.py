"""Base classes for environment implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.env.client import Client
from hud.env.remote_client import RemoteClient
from hud.task import Task
from hud.utils.common import HudStyleConfig, HudStyleConfigs
from hud.utils.config import REMOTE_EVALUATE, REMOTE_SETUP, REMOTE_FUNCTION_PREFIX, expand_config

from hud.adapters.common import CLA

logger = logging.getLogger("hud.environment")


class Observation(BaseModel):
    """
    Observation from the environment.

    Attributes:
        screenshot: Base64 encoded PNG string of the screen
        text: Text observation, if available
    """

    screenshot: str | None = None  # base64 string png
    text: str | None = None


class Environment(BaseModel):
    """
    Environment base class that provides common functionality for all environment implementations.
    This class uses the primitives provided by EnvClient to implement core environment operations.
    """

    metadata: dict[str, Any]
    client: Client
    url: str | None = None
    live_url: str | None = None
    # The task id to use for the environment reset
    task: Task | None = None
    build_data: dict[str, Any]

    # final response
    final_response: str | None = None

    async def _invoke_all(self, configs: HudStyleConfigs) -> list[Any]:
        # Execute each config and collect results
        configs_all = [configs] if not isinstance(configs, list) else configs
        results = []
        for config in configs_all:
            for expanded_config in expand_config(config):
                result, stdout, stderr = await self.client.invoke(expanded_config)
                results.append(result)
                if stdout:
                    logger.info(
                        "%s produced stdout:\n%s",
                        expanded_config.function,
                        stdout.decode(),
                    )
                if stderr:
                    logger.warning(
                        "%s produced stderr:\n%s",
                        expanded_config.function,
                        stderr.decode(),
                    )
        return results
    
    async def _setup(self, config: HudStyleConfigs | None = None) -> None:
        """
        Setup the environment.

        Args:
            config: The configuration to use for the setup
        """
        if isinstance(self.client, RemoteClient):
            await self._invoke_all(create_remote_config(self, config, REMOTE_SETUP))
        else:
            if config is not None:
                await self._invoke_all(config)
            elif self.task and self.task.config is not None:
                await self._invoke_all(self.task.config)
            else:
                raise ValueError("No config or task provided for local environment")

    async def evaluate(self, config: HudStyleConfigs | None = None) -> Any:
        """
        Evaluate the environment.

        Args:
            config: The configuration to use for the evaluation

        Returns:
            Any: Result of the evaluation
        """
        if isinstance(self.client, RemoteClient):
            results = await self._invoke_all(
                create_remote_config(self, config, REMOTE_EVALUATE))
        else:
            if config is not None:
                results = await self._invoke_all(config)
            elif self.task and self.task.config is not None:
                results = await self._invoke_all(self.task.config)
            else:
                raise ValueError("No config or task provided for local environment")
        if len(results) == 1:
            return results[0]
        else:
            return results
        

    async def reset(self, configs: HudStyleConfigs | None = None) -> tuple[
        Observation, dict[str, Any]
    ]:
        """
        Reset the environment.

        Args:
            configs: The configuration to use for the reset

        Returns:
            Observation: The first observation from the environment
            info: Dictionary of information about the environment
        """
        #await self._setup(configs)
        obs, _, _, info = await self.step()
        if self.task and self.task.prompt:
            obs.text = self.task.prompt
        return obs, info

    async def step(self, actions: list[CLA] | None = None) -> tuple[
        Observation, float, bool, dict[str, Any]
    ]:
        """Execute a step in the environment.

        Args:
            action: The action to execute

        Returns:
            Any: Result of the step execution
        """
        if actions is None or len(actions) == 0:
            actions = []
        args = [[action.model_dump() for action in actions]]

        # TODO: Move this into the server side
        if self._maybe_store_response(actions):
            return Observation(text=self.final_response), 0, False, {}
        
        result, stdout, stderr = await self.client.invoke(
            HudStyleConfig(function="step", args=args)
        )
        if stdout:
            logger.info("Step produced stdout: %s", stdout.decode())
        if stderr:
            logger.warning("Step produced stderr: %s", stderr.decode())


        observation = Observation.model_validate(result["observation"], strict=True)

        return observation, 0, False, {}
    
    def _maybe_store_response(self, actions: list[CLA]) -> bool:
        """Store the final response into the environment.

        Args:
            actions: The action(s) to check

        Returns:
            bool: True if the response was submitted, False otherwise
        """
        if len(actions) > 0 and actions[-1].type == "response":
            self.final_response = actions[-1].text
            return True
        return False


    async def get_urls(self) -> dict[str, Any]:
        """Get URLs for the environment.

        Returns:
            dict: Dictionary of URLs for accessing the environment
        """
        data, _, _ = await self.client.invoke(HudStyleConfig(function="get_urls", args=[]))

        self.url = data.get("url")
        self.live_url = data.get("live_url")

        return {
            "url": self.url,
            "live_url": self.live_url,
        }

    async def close(self) -> None:
        """Close the environment.

        This should release any resources and clean up the environment.
        """
        await self.client.close()

def create_remote_config(
    env: Environment | None = None,
    config: HudStyleConfigs | None = None,
    function: str | None = None,
) -> list[HudStyleConfig]:
    """
    Create a configuration based on provided inputs.
    
    Args:
        env: Environment object with optional task configuration
        config: Direct configuration (expanded or not)
        function: Function name to use
        
    Returns:
        list[HudStyleConfig]: List of standardized configurations
        
    Logic:
        1) If explicit config: expand and return HudStyleConfig with func of the function,
        and args of expanded config
        2) If task has the specified function defined: use that
        3) If no task function: check for task._config and use that
        4) If no _config: use task.id and create private_[function]
    """
    # If no function provided, just expand the config and return it directly
    if function is None:
        if config:
            return expand_config(config)
        raise ValueError("Either function or config must be provided")
    
    # Case 1: Explicit config provided
    if config:
        expanded_configs = expand_config(config)
        if env and env.final_response:
            expanded_configs[0].args.append(env.final_response) # for remote responses
        return [HudStyleConfig(function=function, args=expanded_configs)]
    
    # Otherwise, use the environment's task
    task = env.task if env else None
    
    # Must have a task for the remaining cases
    if task is None:
        raise ValueError("Either task or config must be provided")
    
    # Case 2: Task has the specified function attribute
    task_config = getattr(task, function, None)
    if task_config and len(task_config) > 0:
        expanded_configs = expand_config(task_config)
        if task.id:
            expanded_configs[0].id = task.id # for remote IDs
        elif env and env.final_response:
            expanded_configs[0].args.append(env.final_response) # for remote responses
        return [HudStyleConfig(function=function, args=expanded_configs)]
    
    # Case 3: Check for _config
    if hasattr(task, "config") and task.config:
        if task.id:
            task.config["id"] = task.id # for remote IDs
        elif env and env.final_response:
            task.config["args"].append(env.final_response) # for remote responses
        return [HudStyleConfig(function=function, args=[task.config])]
    
    # Case 4: Use task.id
    if task.id:
        return [HudStyleConfig(function=f"{REMOTE_FUNCTION_PREFIX}{function}", args=[task.id])]
    
    # No valid configuration found
    #logger.warning("No valid configuration found for function: %s", function)
    return [HudStyleConfig(function=function, args=[])]

