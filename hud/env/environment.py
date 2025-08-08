"""Base classes for environment implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.env.client import Client
from hud.env.remote_client import RemoteClient, SetupRequest
from hud.task import Task
from hud.utils.agent import format_agent_prompt
from hud.utils.common import FunctionConfig, FunctionConfigs, Observation
from hud.utils.config import (
    LOCAL_EVALUATORS,
    REMOTE_EVALUATE,
    REMOTE_FUNCTION_PREFIX,
    expand_config,
)
from hud.utils.telemetry import stream

logger = logging.getLogger("hud.environment")

if TYPE_CHECKING:
    from hud.adapters.common import CLA
    from hud.agent import Agent


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

    # The task run id
    task_run_id: str | None = None

    # final response
    final_response: str | None = None

    # environment prompt information
    environment_prompt: str | None = None

    async def _invoke_all(self, configs: FunctionConfigs) -> list[Any]:
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

    async def _setup(self, config: FunctionConfigs | None = None) -> None:
        """
        Setup the environment.
        No-op if no config or task is provided.

        Args:
            config: The configuration to use for the setup
        """
        if isinstance(self.client, RemoteClient):
            await self.get_urls()

            setup_request = SetupRequest()

            if self.task:
                setup_request.task_id = self.task.id
                setup_request.config = self.task.config
                setup_request.metadata = _format_task_metadata(self.task)
                if self.task.setup:
                    setup_request.setup = expand_config(self.task.setup)[0]
            elif config:
                setup_request.setup = expand_config(config)[0]
            else:
                raise ValueError("No task or config provided for remote environment")

            result = await self.client.setup(setup_request)

            if result and result.get("id"):
                self.task_run_id = result.get("id")
                logger.info("View the live trace at https://app.hud.so/trace/%s", self.task_run_id)
            else:
                logger.warning("No task run id found in the result")
        else:
            if config is not None:
                await self._invoke_all(config)
            elif self.task and self.task.setup is not None:
                await self._invoke_all(self.task.setup)

    async def evaluate(
        self,
        config: FunctionConfigs | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Evaluate the environment.

        Args:
            config: The configuration to use for the evaluation

        Returns:
            Any: Result of the evaluation
        """
        if isinstance(self.client, RemoteClient):
            results = await self._invoke_all(
                create_remote_config(self, config, REMOTE_EVALUATE, metadata)
            )
        else:
            if config is not None:
                results = await self._invoke_all(config)
            elif self.task and self.task.evaluate is not None:
                results = await self._invoke_all(self.task.evaluate)
            else:
                raise ValueError("No config or task provided for local environment")
        if len(results) == 1:
            return results[0]
        else:
            return results

    async def reset(self) -> tuple[Observation, dict[str, Any]]:
        """
        Reset the environment and return the first observation with the agent prompt.

        Args:
            None

        Returns:
            Observation: The first observation from the environment with the agent prompt
            info: Dictionary of information about the environment
        """
        # await self._setup(configs)
        obs, _, _, info = await self.step()

        if self.build_data.get("environment_prompt"):
            self.environment_prompt = self.build_data["environment_prompt"]

        # Format the agent prompt with the environment prompt and the task prompt
        obs.text = format_agent_prompt(self.environment_prompt, self.task)

        return obs, info

    async def step(
        self,
        actions: CLA | list[CLA] | None = None,
        verbose: bool = False,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute a step in the environment.

        Args:
            action: The action to execute

        Returns:
            Any: Result of the step execution
        """
        if not isinstance(actions, list) and actions is not None:
            actions = [actions]
        if actions is None or len(actions) == 0:
            actions = []
        args = [[action.model_dump() for action in actions]]

        # TODO: Move this into the server side
        self._maybe_store_response(actions)

        result, stdout, stderr = await self.client.invoke(
            FunctionConfig(function="step", args=args)
        )
        if verbose:
            if stdout:
                logger.info("Step produced stdout: %s", stdout.decode())
            if stderr:
                logger.warning("Step produced stderr: %s", stderr.decode())

        observation = Observation.model_validate(result["observation"], strict=True)

        info = result.get("info", {})

        return observation, 0, False, info

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
        data, _, _ = await self.client.invoke(FunctionConfig(function="get_urls", args=[]))

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

    async def stream(self) -> str | None:
        if not self.live_url:
            await self.get_urls()
        if self.live_url is None:
            logger.warning("No live URL found")
            return None
        return stream(self.live_url)

    async def run(self, agent: Agent, max_steps: int = 27, verbose: bool = True) -> Any:
        """Run an agent in the environment.

        Args:
            agent: The agent to run
        """
        if verbose:
            logger.info("Running agent in environment...")
        obs, _ = await self.reset()
        for i in range(max_steps):
            action, done = await agent.predict(obs, verbose=verbose)
            if verbose:
                logger.info(
                    "Step %d: Action: %s",
                    i,
                    [str(a) for a in action] if len(action) > 1 else str(action[0]),
                )
            obs, reward, terminated, info = await self.step(action)
            if verbose:
                logger.info("Step %d: Observation: %s", i, obs)
            if done or terminated:
                break
        result = await self.evaluate()
        if verbose:
            logger.info("Evaluation result: %s", result)
        return result


def _format_task_metadata(task: Task) -> dict[str, Any]:
    metadata = {}
    if task.metadata:
        for key, value in task.metadata.items():
            metadata[str(key)] = value
    if task.sensitive_data:
        metadata["sensitive_data"] = task.sensitive_data
    return metadata


def create_remote_config(
    env: Environment | None = None,
    config: FunctionConfigs | None = None,
    function: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[FunctionConfig]:
    """
    Create a remote configuration for setup or evaluate, determining the final
    function call structure based on the provided task or explicit config.

    This function orchestrates how setup and evaluate steps defined in a Task
    or passed directly are prepared for remote execution via `env._invoke_all`.

    Args:
        env: Environment object, potentially containing a task definition.
             Used to access `env.task` and `env.final_response`.
        config: Direct configuration override (e.g., passed to `env.evaluate(config=...)`).
                Can be in various FunctionConfigs formats.
        function: The top-level function context, typically "setup" or "evaluate".

    Returns:
        list[FunctionConfig]: A list containing a single FunctionConfig object
                              ready for remote invocation via `client.invoke`.
                              The specific function/arguments are chosen based on this priority:
                              1. Explicit `config` parameter (if provided).
                              2. Specific `task` attribute (e.g., `task.evaluate`).
                              3. General `task.config` dictionary.
                              4. Default private function using `task.id`
                              (e.g., `private_evaluate(task.id)`).
                              5. Base `function` name with minimal/default arguments.

    Logic & Examples (Assuming `function="evaluate"` for examples):

        1) Explicit `config` provided: The `config` is expanded and becomes the `args`
           for the top-level `function` call. If the environment has a final_response,
           it's appended to these args.
           - Example Input:
             `env` (with `final_response="Paris"`)
             `config=("contains_text", "Paris")`
             `function="evaluate"`
           - Example Output:
             `[FunctionConfig(function='evaluate', args=[
                FunctionConfig(function='contains_text', args=['Paris', 'Paris'])
             ])]`

        2) No explicit `config`, Task has the attribute (e.g., `task.evaluate`):
           The Task's attribute value (e.g., `task.evaluate`) is expanded and becomes the `args`
           for the top-level `function` call. Task ID is added if present. `final_response` is
           appended if present.
           - Example Input:
             `env` (`task=Task(id="t1", evaluate=("check_answer",), ...)`, `final_response="42"`)
             `config=None`
             `function="evaluate"`
           - Example Output:
             `[FunctionConfig(function='evaluate', args=[FunctionConfig(function='check_answer',
                args=['42'], id='t1')])]`

        3) No explicit `config`, no specific Task attribute, Task has `task.config`:
           The `task.config` dictionary becomes the single argument for the top-level
           `function` call. Task ID is added to the config dict if present. `final_response` is
           appended if present.
           - Example Input:
             `env` (with `task=Task(id="t2", config={"expected": "val"}, ...)`)
             `config=None`
             `function="evaluate"`
           - Example Output:
             `[FunctionConfig(function='evaluate', args=[{"expected": "val", "id": "t2"}])]`

        4) No explicit `config`, no specific Task attribute, no `task.config`, Task has `task.id`:
           Calls a private function (`private_<function>`) on the remote end, passing
           the `task.id` as the only argument.
           - Example Input:
             `env` (with `task=Task(id="t3", ...)`)
             `config=None`
             `function="evaluate"`
           - Example Output:
             `[FunctionConfig(function='private_evaluate', args=['t3'])]`

        5) No explicit `config` and no relevant Task info:
           Calls the top-level `function` with empty args.
           - Example Input:
             `env` (with `task=Task(...)`)
             `config=None`
             `function="evaluate"`
           - Example Output:
             `[FunctionConfig(function='evaluate', args=[])]`
    """
    # If no function provided, just expand the config and return it directly
    if metadata is None:
        metadata = {}
    if function is None:
        if config:
            return expand_config(config)
        raise ValueError("Either function or config must be provided")

    # Case 1: Explicit config provided
    if config:
        if not isinstance(config, dict):
            expanded_configs = expand_config(config)
            if env and env.final_response and expanded_configs[0].args[0] in LOCAL_EVALUATORS:
                # Ensure args is a list before appending
                if not isinstance(expanded_configs[0].args, list):
                    expanded_configs[0].args = [expanded_configs[0].args]
                expanded_configs[0].args.append(env.final_response)  # for remote responses
            return [FunctionConfig(function=function, args=expanded_configs, metadata=metadata)]
        else:
            return [FunctionConfig(function=function, args=[config], metadata=metadata)]

    # Otherwise, use the environment's task
    task = env.task if env else None

    # Must have a task for the remaining cases
    if task is None:
        raise ValueError("Either task or config must be provided")

    metadata = _format_task_metadata(task)

    # Case 2: Task has the specified function attribute
    task_config = getattr(task, function, None)
    if task_config:
        expanded_configs = expand_config(task_config)
        if task.id:
            expanded_configs[0].id = task.id  # for remote IDs
        if env and env.final_response and expanded_configs[0].function in LOCAL_EVALUATORS:
            # Ensure args is a list before appending
            if not isinstance(expanded_configs[0].args, list):
                expanded_configs[0].args = [expanded_configs[0].args]
            expanded_configs[0].args.append(env.final_response)  # for remote responses
        return [FunctionConfig(function=function, args=expanded_configs, metadata=metadata)]

    # Case 3: Check for task.config
    if hasattr(task, "config") and task.config:
        # The server expects a dictionary for evaluation requests, but task.config
        # can be either a dict or a list (e.g., OSWorld-Verified uses lists)
        if isinstance(task.config, dict):
            # For dict configs, we can modify it directly
            final_args = task.config.copy()
            if task.id:
                final_args["id"] = task.id  # for remote IDs
            if env and env.final_response:
                # Append response, ensuring args exists and is a list
                if "args" not in final_args:
                    final_args["args"] = []
                if not isinstance(final_args["args"], list):
                    final_args["args"] = [final_args["args"]]
                final_args["args"].append(env.final_response)
        elif isinstance(task.config, list):
            # For list configs (like OSWorld-Verified), wrap in a dict
            # The server expects a dict, so we preserve the list under a 'config' key
            final_args = {"config": task.config}
            if task.id:
                final_args["id"] = task.id
            if env and env.final_response:
                final_args["args"] = [env.final_response]
        else:
            # For other types, convert to dict to ensure server compatibility
            final_args = {"config": task.config}
            if task.id:
                final_args["id"] = task.id
            if env and env.final_response:
                final_args["args"] = [env.final_response]
        
        return [FunctionConfig(function=function, args=[final_args], metadata=metadata)]

    # Case 4: Use task.id
    if task.id:
        args_list = [task.id]
        if env and env.final_response:
            args_list.append(env.final_response)  # Append final response
        return [
            FunctionConfig(
                function=f"{REMOTE_FUNCTION_PREFIX}{function}",
                args=args_list,
                metadata=metadata,
            )
        ]

    # Case 5: No valid configuration found
    args_list = []
    if env and env.final_response:
        args_list.append(env.final_response)
    return [FunctionConfig(function=function, args=args_list, metadata=metadata)]
