from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hud.tools.base import ToolResult

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class BaseExecutor:
    """
    Base executor that simulates actions without executing them.
    Used as a fallback when actual execution is not possible (e.g., headless environments).

    This executor dynamically handles any method call and returns a description
    of what was attempted, making it a perfect simulation fallback.
    """

    def __init__(self, display_num: int | None = None) -> None:
        """
        Initialize the base executor.

        Args:
            display_num: X display number (ignored in base executor)
        """
        self.display_num = display_num
        self._screenshot_delay = 0.5
        self.is_simulation = True
        logger.info("BaseExecutor initialized - running in simulation mode")

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """
        Dynamically handle any method call by returning a simulation function.

        Args:
            name: The method name being called

        Returns:
            A callable that simulates the method execution
        """

        async def simulate_method(*args: Any, **kwargs: Any) -> ToolResult:
            """Simulate any method call with its arguments."""

            # Format arguments for display
            arg_parts = []

            # Add positional arguments
            if args:
                formatted_args = []
                for arg in args:
                    if isinstance(arg, str):
                        formatted_args.append(f"'{arg}'")
                    else:
                        formatted_args.append(str(arg))
                arg_parts.extend(formatted_args)

            # Add keyword arguments
            if kwargs:
                formatted_kwargs = []
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        formatted_kwargs.append(f"{key}='{value}'")
                    else:
                        formatted_kwargs.append(f"{key}={value}")
                arg_parts.extend(formatted_kwargs)

            # Create the simulation message
            args_str = ", ".join(arg_parts) if arg_parts else ""
            simulation_msg = f"[SIMULATED] {name}({args_str}) attempted"

            logger.info(simulation_msg)

            # Check if this method should include a screenshot
            take_screenshot = kwargs.get("take_screenshot", True)
            screenshot_data = None

            # Methods that typically return screenshots or should trigger them
            screenshot_methods = {
                "screenshot",
                "click",
                "type_text",
                "key",
                "key_press",
                "move_mouse",
                "mouse_move",
                "scroll",
                "drag",
                "execute",
            }

            if name in screenshot_methods and take_screenshot:
                screenshot_data = await self.screenshot()

            return ToolResult(output=simulation_msg, base64_image=screenshot_data)

        # Return the simulation function
        return simulate_method

    async def screenshot(self) -> str | None:
        """
        Simulate taking a screenshot.

        Returns:
            Base64 encoded fake screenshot data
        """
        logger.info("[SIMULATION] Taking screenshot")

        # Create a minimal 1x1 PNG image as base64
        # This is a 1x1 transparent PNG
        fake_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # noqa: E501

        return fake_png_base64

    async def execute(self, command: str, take_screenshot: bool = True) -> ToolResult:
        """
        Simulate executing a command.

        Args:
            command: The command to simulate
            take_screenshot: Whether to simulate taking a screenshot

        Returns:
            ToolResult with simulated output
        """
        logger.info("[SIMULATION] Would execute: %s", command)

        # Simulate the command execution
        output = f"[SIMULATED] Command executed: {command}"

        # Simulate screenshot if requested
        screenshot_data = None
        if take_screenshot:
            screenshot_data = await self.screenshot()

        return ToolResult(output=output, base64_image=screenshot_data)
