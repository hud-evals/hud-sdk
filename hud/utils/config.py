from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from hud.utils.common import FunctionConfig, FunctionConfigs

if TYPE_CHECKING:
    from typing import TypeGuard

logger = logging.getLogger("hud.utils.config")

REMOTE_FUNCTION_PREFIX = "private_"
REMOTE_SETUP = "setup"
REMOTE_EVALUATE = "evaluate"

LOCAL_EVALUATORS = ["response_is", "response_includes", "response_match"]


def _is_valid_python_name(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))


def _validate_hud_config(config: dict) -> FunctionConfig:
    """Validate and convert a dictionary to an FunctionConfig."""
    if not isinstance(config.get("function"), str):
        raise ValueError("function must be a string")

    # Validate function path components
    _split_and_validate_path(config["function"])

    args = config["args"] if isinstance(config.get("args"), list) else [config["args"]]

    # Create a proper FunctionConfig object instead of using cast
    return FunctionConfig(function=config["function"], args=args, id=config.get("id"))


def _split_and_validate_path(path: str) -> None:
    """Split a function path into components, validating each part."""
    parts = path.split(".")

    if not parts:
        raise ValueError("Empty function path")

    # Validate each part
    for part in parts:
        if not _is_valid_python_name(part):
            raise ValueError(f"Invalid Python identifier in path: {part}")


def _is_list_of_configs(config: FunctionConfigs) -> TypeGuard[list[FunctionConfig]]:
    """Check if a config is a list of FunctionConfig objects."""
    return isinstance(config, list) and all(isinstance(item, FunctionConfig) for item in config)


def expand_config(config: FunctionConfigs) -> list[FunctionConfig]:
    """
    Process a config into a standardized list of FunctionConfig objects.

    Args:
        config: Can be:
            - A tuple where first element is function name and rest are args
            - A FunctionConfig object
            - A dictionary with "function" and "args" keys
            - A list of FunctionConfig objects

    Returns:
        list[FunctionConfig]: List of standardized configurations

    Raises:
        ValueError: If the configuration format is invalid
    """
    logger.debug("Processing config: %s", config)

    # If it's already a FunctionConfig, just wrap it in a list
    if isinstance(config, FunctionConfig):
        return [config]

    # If it's a list of FunctionConfigs, return as is
    if _is_list_of_configs(config):
        return config

    # Handle dictionary configuration
    if isinstance(config, dict):
        return [_validate_hud_config(config)]

    if isinstance(config, str):
        return [FunctionConfig(function=config, args=[])]

    # Handle tuple format
    if isinstance(config, tuple):
        if len(config) < 1 or not isinstance(config[0], str):
            error_msg = "Invalid tuple configuration. "
            "Expected tuple[str, ...], got: {type(config)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # First element is the function name, rest are args
        function_name = config[0]
        args = list(config[1:]) if len(config) > 1 else []

        return [FunctionConfig(function=function_name, args=args)]

    # Unknown configuration type
    error_msg = f"Unknown configuration type: {type(config)}"
    logger.error(error_msg)
    raise ValueError(error_msg)
