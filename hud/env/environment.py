"""Base classes for environment implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.adapters.common.types import WaitAction
from hud.env.client import Client
from hud.env.remote_client import RemoteClient
from hud.task import Task
from hud.utils import HudStyleConfigs, expand_config
from hud.utils.config import REMOTE_EVALUATE, REMOTE_SETUP, HudStyleConfig, create_remote_config

if TYPE_CHECKING:
    from hud.adapters.common.types import CLA

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
            await self._invoke_all(create_remote_config(self.task, config, REMOTE_SETUP))
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
            results = await self._invoke_all(create_remote_config(self.task, config, REMOTE_EVALUATE))
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
        

    async def reset(self, configs: HudStyleConfigs | None = None) -> tuple[Observation, dict[str, Any]]:
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

    async def step(self, actions: list[CLA] | None = None) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute a step in the environment.

        Args:
            action: The action to execute

        Returns:
            Any: Result of the step execution
        """

        result, stdout, stderr = await self.client.invoke(
            HudStyleConfig(function="step", args=[[action.model_dump() for action in actions] if actions is not None else [WaitAction(time=100).model_dump()]])
        )
        if stdout:
            logger.info("Step produced stdout: %s", stdout.decode())
        if stderr:
            logger.warning("Step produced stderr: %s", stderr.decode())


        observation = Observation.model_validate(result["observation"], strict=True)

        return observation, 0, False, {}

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
