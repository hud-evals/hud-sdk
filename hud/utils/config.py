from __future__ import annotations
import logging
from typing import Any, TypedDict, Union

logger = logging.getLogger("hud.utils.config")

class ExpandedConfig(TypedDict):
    function: list[str]
    args: list[Any]

class SemiExpandedConfig(TypedDict):
    # function can be a str (x.y.foo) or a list of str (["x", "y", "foo"])
    function: Union[str, list[str]]
    args: list[Any]

HudStyleConfig = Union[str, SemiExpandedConfig, list[SemiExpandedConfig]]

def expand_config(config: HudStyleConfig) -> list[ExpandedConfig]:
    """
    Process a configuration into a standardized list of dictionary formats.
    
    Args:
        config: The configuration, which can be:
            - String (function name): "chrome.maximize"
            - String (function with args): "chrome.activate_tab 5"
            - Dict: {"function": "function_name", "args": [...]}
            - List of the above
            
    Returns:
        list[ExpandedConfig]: List of standardized configurations with function and args
        
    Raises:
        ValueError: If the configuration format is not recognized
    """
    logger.debug("Processing config: %s", config)
    
    # Handle list of configurations directly
    if isinstance(config, list):
        results = []
        for item in config:
            # Recursively process each item and extend results
            # (process_config always returns a list)
            results.extend(expand_config(item))
        return results
    
    # Handle string configurations
    if isinstance(config, str):
        # Check if it's a simple function name or function with args
        parts = config.split(maxsplit=1)
        if len(parts) == 1:
            # Just a function name, no args
            return [{"function": parts[0].split("."), "args": []}]
        else:
            # Function with space-separated args
            func_name, args_str = parts
            return [{"function": func_name.split("."), "args": [args_str]}]
            
    # Handle dictionary configurations
    elif isinstance(config, dict) and "function" in config:
        func_name = config["function"]
        args = config.get("args", [])
        
        if not isinstance(args, list):
            # Single arg, convert to list
            args = [args]
        
        if isinstance(func_name, str):
            func_name = func_name.split(".")

        return [{"function": func_name, "args": args}]
        
    # Unknown configuration type
    error_msg = f"Unknown configuration type: {type(config)}"
    logger.error(error_msg)
    raise ValueError(error_msg)
