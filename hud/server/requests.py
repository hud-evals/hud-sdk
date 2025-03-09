"""
HTTP request utilities for the HUD API.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("hud.http")


class RequestError(Exception):
    """Custom exception for API request errors"""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_text: str | None = None,
        response_json: dict[str, Any] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        self.response_json = response_json
        self.response_headers = response_headers
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]

        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        if self.response_text:
            parts.append(f"Response Text: {self.response_text}")
        if self.response_json:
            parts.append(f"Response JSON: {self.response_json}")
        if self.response_headers:
            parts.append(f"Headers: {self.response_headers}")

        return " | ".join(parts)

    @classmethod
    def from_http_error(cls, error: httpx.HTTPStatusError) -> RequestError:
        """Create a RequestError from an HTTP error response"""
        response = error.response
        status_code = response.status_code
        response_text = response.text
        response_headers = dict(response.headers)

        # Try to get detailed error info from JSON if available
        response_json = None
        try:
            response_json = response.json()
            detail = response_json.get("detail")
            if detail:
                message = f"Request failed: {detail}"
            else:
                # If no detail field but we have JSON, include a summary
                message = f"Request failed with status {status_code}"
                if len(response_json) <= 5:  # If it's a small object, include it in the message
                    message += f" - JSON response: {response_json}"
        except Exception:
            # Fallback to simple message if JSON parsing fails
            message = f"Request failed with status {status_code}"

        # Log the error details
        logger.error(
            "HTTP error from HUD SDK: %s | URL: %s | Status: %s | Response: %s%s",
            message,
            response.url,
            status_code,
            response_text[:500],
            "..." if len(response_text) > 500 else "",
        )

        return cls(
            message=message,
            status_code=status_code,
            response_text=response_text,
            response_json=response_json,
            response_headers=response_headers,
        )


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
            result = response.json()
            return result
        except httpx.HTTPStatusError as e:
            raise RequestError.from_http_error(e) from None
        except httpx.RequestError as e:
            raise RequestError(f"Network error: {e!s}") from None
        except Exception as e:
            # Catch-all for unexpected errors
            raise RequestError(f"Unexpected error: {e!s}") from None


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
            result = response.json()
            return result
        except httpx.HTTPStatusError as e:
            raise RequestError.from_http_error(e) from None
        except httpx.RequestError as e:
            raise RequestError(f"Network error: {e!s}") from None
        except Exception as e:
            # Catch-all for unexpected errors
            raise RequestError(f"Unexpected error: {e!s}") from None
