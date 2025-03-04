"""
HTTP request utilities for the HUD API.
"""

from __future__ import annotations

from typing import Any

import httpx


class RequestError(Exception):
    """
    Custom exception for API request errors.
    """


async def make_request(
    method: str, url: str, json: Any | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """
    Make an asynchronous HTTP request to the HUD API.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL for the request
        json: Optional JSON serializable data
        api_key: API key for authentication
        
    Returns:
        dict: JSON response from the server
        
    Raises:
        RequestError: If API key is missing or request fails
    """
    if not api_key:
        raise RequestError("API key is required but not provided")

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=240.0) as client:
        try:
            response = await client.request(method=method, url=url, json=json, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise RequestError(f"Request failed: {e!s}") from None


def make_sync_request(
    method: str, url: str, json: Any | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """
    Make a synchronous HTTP request to the HUD API.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL for the request
        json: Optional JSON serializable data
        api_key: API key for authentication
        
    Returns:
        dict: JSON response from the server
        
    Raises:
        RequestError: If API key is missing or request fails
    """
    if not api_key:
        raise RequestError("API key is required but not provided")

    headers = {"Authorization": f"Bearer {api_key}"}

    with httpx.Client(timeout=240.0) as client:
        try:
            response = client.request(method=method, url=url, json=json, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise RequestError(f"Request failed: {e!s}") from None
