"""
HUD client for interacting with the API.
"""

from __future__ import annotations

import json
from typing import Any

from .adapters.common import Adapter
from .evalset import EvalSet
from .run import Run, RunResponse
from .server import make_request
from .settings import settings


class HUDClient:
    """
    Client for interacting with the HUD API.

    This is the main entry point for the SDK, providing methods to load gyms,
    evalsets, and create runs.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the HUD client with an API key.

        Args:
            api_key: API key for authentication with the HUD API
        """
        self.api_key = api_key or settings.api_key
        settings.api_key = self.api_key

    def display_stream(self, live_url: str | None = None) -> None:
        """
        Display a stream in the HUD system.
        """
        if live_url is None:
            raise ValueError("live_url cannot be None")
        from IPython.display import HTML, display

        html_content = f"""
        <div style="width: 960px; height: 540px; overflow: hidden;">
            <div style="transform: scale(0.5); transform-origin: top left;">
                <iframe src="{live_url}" width="1920" height="1080" style="border: 1px solid #ddd;">
                </iframe>
            </div>
        </div>
        """
        display(HTML(html_content))
