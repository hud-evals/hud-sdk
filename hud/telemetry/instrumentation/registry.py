from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class InstrumentorRegistry:
    """Registry for telemetry instrumentors."""

    def __init__(self) -> None:
        self._instrumentors: dict[str, Any] = {}

    def register(self, name: str, instrumentor: Any) -> None:
        """Register an instrumentor.

        Args:
            name: Name of the instrumentor
            instrumentor: The instrumentor instance
        """
        self._instrumentors[name] = instrumentor
        logger.debug("Registered instrumentor: %s", name)

    def _safe_install(self, name: str, instrumentor: Any) -> tuple[str, Exception | None]:
        """Safely install an instrumentor and return result."""
        try:
            instrumentor.install()
            return name, None
        except Exception as e:
            return name, e

    def install_all(self) -> None:
        """Install all registered instrumentors."""
        # Use map to apply safe installation to all instrumentors
        installation_results = [
            self._safe_install(name, instrumentor)
            for name, instrumentor in self._instrumentors.items()
        ]

        # Process results
        for name, error in installation_results:
            if error is None:
                logger.debug("Installed instrumentor: %s", name)
            else:
                logger.warning("Failed to install instrumentor %s: %s", name, error)


# Create a singleton registry
registry = InstrumentorRegistry()

# Try to register MCP instrumentor if available
try:
    from .mcp import MCPInstrumentor

    registry.register("mcp", MCPInstrumentor())
    logger.debug("MCP instrumentor registered")
except Exception as e:
    logger.debug("Could not register MCP instrumentor: %s", e)
