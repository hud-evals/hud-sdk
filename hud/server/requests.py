"""
HTTP request utilities for the HUD API.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

# Set up logger
logger = logging.getLogger("hud.http")
logger.setLevel(logging.DEBUG)


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
    def from_http_error(
        cls, error: httpx.HTTPStatusError, context: str = ""
    ) -> RequestError:
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
                if (
                    len(response_json) <= 5
                ):  # If it's a small object, include it in the message
                    message += f" - JSON response: {response_json}"
        except Exception:
            # Fallback to simple message if JSON parsing fails
            message = f"Request failed with status {status_code}"

        # Add context if provided
        if context:
            message = f"{context}: {message}"

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


async def _handle_retry(
    attempt: int, max_retries: int, retry_delay: float, url: str, error_msg: str
) -> None:
    """Helper function to handle retry logic and logging."""
    retry_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
    logger.warning(
        "%s from %s, retrying in %.2f seconds (attempt %d/%d)",
        error_msg,
        url,
        retry_time,
        attempt,
        max_retries,
    )
    await asyncio.sleep(retry_time)


async def make_request(
    method: str,
    url: str,
    json: Any | None = None,
    api_key: str | None = None,
    max_retries: int = 4,
    retry_delay: float = 2.0,
) -> dict[str, Any]:
    """
    Make an asynchronous HTTP request to the HUD API.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL for the request
        json: Optional JSON serializable data
        api_key: API key for authentication
        max_retries: Maximum number of retries
        retry_delay: Delay between retries
    Returns:
        dict: JSON response from the server

    Raises:
        RequestError: If API key is missing or request fails
    """
    if not api_key:
        raise RequestError("API key is required but not provided")

    headers = {"Authorization": f"Bearer {api_key}"}
    retry_status_codes = [502, 503, 504]
    attempt = 0

    while attempt <= max_retries:
        attempt += 1

        try:
            async with httpx.AsyncClient(
                timeout=600.0, # Long running requests can take up to 10 minutes
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=1000,
                    keepalive_expiry=10.0,
                ),
            ) as client:
                response = await client.request(
                    method=method, url=url, json=json, headers=headers
                )

            # Check if we got a retriable status code
            if response.status_code in retry_status_codes and attempt <= max_retries:
                await _handle_retry(
                    attempt,
                    max_retries,
                    retry_delay,
                    url,
                    f"Received status {response.status_code}",
                )
                continue

            response.raise_for_status()
            result = response.json()
            return result
        except httpx.HTTPStatusError as e:
            raise RequestError.from_http_error(e) from None
        except httpx.RequestError as e:
            if attempt <= max_retries:
                await _handle_retry(
                    attempt, max_retries, retry_delay, url, f"Network error: {e}"
                )
                continue
            else:
                raise RequestError(f"Network error: {e!s}") from None
        except Exception as e:
            raise RequestError(f"Unexpected error: {e!s}") from None
    raise RequestError(f"Request failed after {max_retries} retries with unknown error")


def make_request_sync(
    method: str,
    url: str,
    json: Any | None = None,
    api_key: str | None = None,
    max_retries: int = 4,
    retry_delay: float = 2.0,
) -> dict[str, Any]:
    """
    Make a synchronous HTTP request to the HUD API.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL for the request
        json: Optional JSON serializable data
        api_key: API key for authentication
        max_retries: Maximum number of retries
        retry_delay: Delay between retries
    Returns:
        dict: JSON response from the server

    Raises:
        RequestError: If API key is missing or request fails
    """
    if not api_key:
        raise RequestError("API key is required but not provided")

    headers = {"Authorization": f"Bearer {api_key}"}
    retry_status_codes = [502, 503, 504]
    attempt = 0

    while attempt <= max_retries:
        attempt += 1

        try:
            with httpx.Client(
                timeout=600.0, # Long running requests can take up to 10 minutes
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=1000,
                    keepalive_expiry=10.0,
                ),
            ) as client:
                response = client.request(
                    method=method, url=url, json=json, headers=headers
                )

            # Check if we got a retriable status code
            if response.status_code in retry_status_codes and attempt <= max_retries:
                retry_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    "Received status %d from %s, retrying in %.2f seconds (attempt %d/%d)",
                    response.status_code,
                    url,
                    retry_time,
                    attempt,
                    max_retries,
                )
                time.sleep(retry_time)
                continue

            response.raise_for_status()
            result = response.json()
            return result
        except httpx.HTTPStatusError as e:
            raise RequestError.from_http_error(e) from None
        except httpx.RequestError as e:
            if attempt <= max_retries:
                retry_time = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Network error %s from %s, retrying in %.2f seconds (attempt %d/%d)",
                    str(e),
                    url,
                    retry_time,
                    attempt,
                    max_retries,
                )
                time.sleep(retry_time)
                continue
            else:
                raise RequestError(f"Network error: {e!s}") from None
        except Exception as e:
            raise RequestError(f"Unexpected error: {e!s}") from None
    raise RequestError(f"Request failed after {max_retries} retries with unknown error")
