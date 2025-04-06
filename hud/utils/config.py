from __future__ import annotations
import logging
import re
from typing import Any, TypedDict, Union, cast, Dict
from lark import Lark, Transformer

logger = logging.getLogger("hud.utils.config")

# Grammar for parsing function calls
FUNC_CALL_GRAMMAR = r"""
    ?start: funccall

    funccall: funcname "(" [funcargs] ")"
    funcname: CNAME ("." CNAME)*
    funcargs: funcarg ("," funcarg)* [","]
    funcarg: NUMBER -> number
           | STRING -> string
           | "[" [funcargs] "]" -> list
           | funccall -> nested_call

    %import common.CNAME
    %import common.NUMBER
    %import common.WS
    %import common.ESCAPED_STRING -> STRING
    %ignore WS
"""

class FuncCallTransformer(Transformer):
    def funccall(self, items):
        funcname = items[0]
        args = items[1] if len(items) > 1 else []
        return {"module": funcname[:-1], "function": funcname[-1], "args": args}
    
    def funcname(self, items):
        return [str(token) for token in items]
    
    def funcargs(self, items):
        return list(items)
    
    def number(self, items):
        return float(items[0].value)
    
    def string(self, items):
        return items[0].value[1:-1]  # Remove quotes
    
    def list(self, items):
        return items[0] if items else []
    
    def nested_call(self, items):
        return items[0]
    
    def CNAME(self, token):
        return str(token)

# Create the parser
func_call_parser = Lark(FUNC_CALL_GRAMMAR, parser='lalr', transformer=FuncCallTransformer())

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
    """Process a string configuration into ExpandedConfig format.
    
    The string should be in the format of a function call, e.g.:
    - "chrome.maximize()"
    - "browser.open('https://example.com')"
    - "app.set_position([100, 200])"
    """
    result = func_call_parser.parse(config)
    return [cast(ExpandedConfig, result)]

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
            - String (function name): "chrome.maximize()"
            - String (function with args): "chrome.activate_tab(5)"
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
