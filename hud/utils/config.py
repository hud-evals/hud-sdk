from __future__ import annotations
import logging
import re
from typing import Any, TypedDict, Union, cast, Dict

logger = logging.getLogger("hud.utils.config")

class ExpandedConfig(TypedDict):
    module: list[str]
    function: str
    args: list[Any]

class SemiExpandedConfig(TypedDict):
    # function can be a str (x.y.foo) or a list of str (["x", "y", "foo"])
    function: Union[str, list[str]]
    args: list[Any]

HudStyleConfig = Union[str, SemiExpandedConfig, ExpandedConfig, list[str],list[SemiExpandedConfig], list[ExpandedConfig]]

def _is_valid_python_name(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))

def _validate_expanded_config(config: Dict[str, Any]) -> ExpandedConfig:
    """Validate and convert a dictionary to an ExpandedConfig."""
    if not isinstance(config.get("module"), list):
        raise ValueError("module must be a list")
    if not isinstance(config.get("function"), str):
        raise ValueError("function must be a string")
    if not isinstance(config.get("args"), list):
        raise ValueError("args must be a list")
    
    # Validate module path components
    for part in config["module"]:
        if not _is_valid_python_name(part):
            raise ValueError(f"Invalid Python identifier in module path: {part}")
    
    # Validate function name
    if not _is_valid_python_name(config["function"]):
        raise ValueError(f"Invalid Python identifier for function: {config['function']}")
    
    return cast(ExpandedConfig, config)

def _split_and_validate_path(path: Union[str, list[str]]) -> tuple[list[str], str]:
    """Split a function path into module and function parts, validating each component."""
    if isinstance(path, str):
        parts = path.split(".")
    else:
        parts = path
    
    if not parts:
        raise ValueError("Empty function path")
    
    # Validate each part
    for part in parts:
        if not _is_valid_python_name(part):
            raise ValueError(f"Invalid Python identifier in path: {part}")
    
    # Last part is the function name, rest is module path
    return parts[:-1], parts[-1]

def _process_string_config(config: str) -> list[ExpandedConfig]:
    """Process a string configuration into ExpandedConfig format."""
    parts = config.split(maxsplit=1)
    if len(parts) == 1:
        # Just a function name, no args
        module, function = _split_and_validate_path(parts[0])
        return [{"module": module, "function": function, "args": []}]
    else:
        # Function with space-separated args
        func_name, args_str = parts
        module, function = _split_and_validate_path(func_name)
        return [{"module": module, "function": function, "args": [args_str]}]

def _process_dict_config(config: Union[SemiExpandedConfig, ExpandedConfig]) -> list[ExpandedConfig]:
    """Process a dictionary configuration into ExpandedConfig format."""
    # Convert to regular dict for easier handling
    config_dict = dict(config)
    
    if "module" in config_dict and "function" in config_dict:
        # Already in ExpandedConfig format, just validate
        return [_validate_expanded_config(config_dict)]
    
    # SemiExpandedConfig format
    if "function" not in config_dict:
        raise ValueError("Configuration must contain 'function' key")
    
    func_name = config_dict["function"]
    if not isinstance(func_name, (str, list)):
        raise ValueError("function must be either a string or a list of strings")
    
    args = config_dict.get("args", [])
    if not isinstance(args, list):
        args = [args]
    
    module, function = _split_and_validate_path(cast(Union[str, list[str]], func_name))
    return [{"module": module, "function": function, "args": args}]

def expand_config(config: HudStyleConfig) -> list[ExpandedConfig]:
    """
    Process a configuration into a standardized list of ExpandedConfig formats.
    
    Args:
        config: The configuration, which can be:
            - String (function name): "chrome.maximize"
            - String (function with args): "chrome.activate_tab 5"
            - SemiExpandedConfig: {"function": "chrome.maximize", "args": []}
            - ExpandedConfig: {"module": ["chrome"], "function": "maximize", "args": []}
            - List of any of the above
            
    Returns:
        list[ExpandedConfig]: List of standardized configurations with module, function and args
        
    Raises:
        ValueError: If the configuration format is not recognized or contains invalid Python identifiers
    """
    logger.debug("Processing config: %s", config)
    
    # Handle list of configurations
    if isinstance(config, list):
        results = []
        for item in config:
            results.extend(expand_config(item))
        return results
    
    # Handle string configurations
    if isinstance(config, str):
        return _process_string_config(config)
    
    # Handle dictionary configurations
    if isinstance(config, dict):
        return _process_dict_config(config)
    
    # Unknown configuration type
    error_msg = f"Unknown configuration type: {type(config)}"
    logger.error(error_msg)
    raise ValueError(error_msg)
