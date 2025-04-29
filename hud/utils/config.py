from __future__ import annotations

import logging
import re

from hud.utils.common import HudStyleConfig, HudStyleConfigs

logger = logging.getLogger("hud.utils.config")

REMOTE_FUNCTION_PREFIX = "private_"
REMOTE_SETUP = "setup"
REMOTE_EVALUATE = "evaluate"

def _is_valid_python_name(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))

def _validate_hud_config(config: dict) -> HudStyleConfig:
    """Validate and convert a dictionary to an HudStyleConfig."""
    if not isinstance(config.get("function"), str):
        raise ValueError("function must be a string")
    
    # Validate function path components
    _split_and_validate_path(config["function"])

    args = config["args"] if isinstance(config.get("args"), list) else [config["args"]]
    
    # Create a proper HudStyleConfig object instead of using cast
    return HudStyleConfig(function=config["function"], args=args, id=config.get("id"))

def _split_and_validate_path(path: str) -> None:
    """Split a function path into components, validating each part."""
    parts = path.split(".")

    if not parts:
        raise ValueError("Empty function path")
    
    # Validate each part
    for part in parts:
        if not _is_valid_python_name(part):
            raise ValueError(f"Invalid Python identifier in path: {part}")

def expand_config(config: HudStyleConfigs) -> list[HudStyleConfig]:
    """
    Process a config into a standardized list of HudStyleConfig objects.
    
    Args:
        config: Can be:
            - A tuple where first element is function name and rest are args
            - A HudStyleConfig object
            - A dictionary with "function" and "args" keys
            - A list of HudStyleConfig objects
            
    Returns:
        list[HudStyleConfig]: List of standardized configurations
        
    Raises:
        ValueError: If the configuration format is invalid
    """
    logger.debug("Processing config: %s", config)

    # If it's already a HudStyleConfig, just wrap it in a list
    if isinstance(config, HudStyleConfig):
        return [config]
    
    # If it's a list of HudStyleConfigs, return as is
    if isinstance(config, list) and all(isinstance(item, HudStyleConfig) for item in config):
        return config # type: ignore
    
    # Handle dictionary configuration
    if isinstance(config, dict):
        return [_validate_hud_config(config)]
    
    if isinstance(config, str):
        return [HudStyleConfig(function=config, args=[])]
    
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
        
        return [HudStyleConfig(function=function_name, args=args)]
    
    # Unknown configuration type
    error_msg = f"Unknown configuration type: {type(config)}"
    logger.error(error_msg)
    raise ValueError(error_msg)
