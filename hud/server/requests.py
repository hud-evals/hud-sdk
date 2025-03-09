"""
HTTP request utilities for the HUD API.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("hud.http")


class RequestError(Exception):
    """Custom exception for API request errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        response_json: Optional[Dict[str, Any]] = None,
        response_headers: Optional[Dict[str, str]] = None,
    ):
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
    def from_http_error(cls, error: httpx.HTTPStatusError, context: str = "") -> "RequestError":
        """Create a RequestError from an HTTP error response"""
        response = error.response
        status_code = response.status_code
        response_text = response.text
        response_headers = dict(response.headers)
        url = str(response.url)

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
        f"{error_msg} from {url}, "
        f"retrying in {retry_time:.2f} seconds (attempt {attempt}/{max_retries})"
    )
    await asyncio.sleep(retry_time)


def _create_request_error(message: str) -> RequestError:
    """Helper function to create a RequestError with consistent parameters."""
    return RequestError(
        message=message,
        status_code=None,
        response_text=None,
    )


async def make_request(
    method: str,
    url: str,
    json: Any | None = None,
    api_key: str | None = None,
    params: Optional[Dict[str, Any]] = None,
    data: Any = None,
    files: Any = None,
    timeout: float = 240.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_status_codes: Optional[List[int]] = None,
) -> dict[str, Any]:
    """
    Make an asynchronous HTTP request to the HUD API.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL for the request
        json: Optional JSON serializable data
        api_key: API key for authentication
        params: Optional query parameters
        data: Optional form data or plain text payload
        files: Optional files to upload
        timeout: Request timeout in seconds (default: 4 minutes)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
                     This will be exponentially increased with each retry
        retry_status_codes: List of HTTP status codes to retry (default: [502, 503, 504])

    Returns:
        dict: JSON response from the server

    Raises:
        RequestError: If API key is missing or request fails
    """
    if not api_key:
        raise RequestError("API key is required but not provided")

    # Initialize parameters
    headers = {"Authorization": f"Bearer {api_key}"}
    retry_status_codes = retry_status_codes or [502, 503, 504]

    # Track attempts
    attempt = 0
    last_exception = None

    while attempt <= max_retries:
        attempt += 1

        # Log the outgoing request with attempt number if retrying
        if attempt == 1:
            logger.info(f"Sending request to {url}")
        else:
            logger.info(f"Retry attempt {attempt - 1}/{max_retries} for request to {url}")

        try:
            # Make the request
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                )

            # Log the response
            logger.info(f"Received response from {url}: Status {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response content length: {len(response.content)}")

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

            # Raise exception for other error status codes
            response.raise_for_status()

            return response.json()

        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code in retry_status_codes and attempt <= max_retries:
                await _handle_retry(
                    attempt, max_retries, retry_delay, url, f"HTTP error {e.response.status_code}"
                )
                continue
            else:
                # Non-retriable error or max retries exceeded
                if attempt > max_retries:
                    logger.error(f"HTTP error after {max_retries} retries: {str(e)}")
                raise RequestError.from_http_error(
                    e, context=f"Error making {method} request to {url}"
                ) from None

        except httpx.RequestError as e:
            last_exception = e
            if attempt <= max_retries:
                await _handle_retry(
                    attempt, max_retries, retry_delay, url, f"Network error: {str(e)}"
                )
                continue
            else:
                error = RequestError(f"Network error: {e!s}")
                logger.error(f"Network error after {max_retries} retries: {str(error)}")
                raise error from None

        except Exception as e:
            last_exception = e
            error = RequestError(f"Unexpected error: {e!s}")
            logger.error(f"Unexpected error during request: {str(error)}")
            raise error from None

    # If we've exhausted all retries
    if last_exception:
        if isinstance(last_exception, httpx.HTTPStatusError):
            raise RequestError.from_http_error(
                last_exception, context=f"Request failed after {max_retries} retries"
            ) from None
        else:
            raise RequestError(
                f"Request failed after {max_retries} retries: {str(last_exception)}"
            ) from None

    # This should never happen, but just in case
    raise RequestError(f"Request failed after {max_retries} retries with unknown error")


def make_sync_request(
    method: str,
    url: str,
    json: Any | None = None,
    api_key: str | None = None,
    params: Optional[Dict[str, Any]] = None,
    data: Any = None,
    files: Any = None,
    timeout: float = 240.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_status_codes: Optional[List[int]] = None,
) -> dict[str, Any]:
    """
    Make a synchronous HTTP request to the HUD API.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL for the request
        json: Optional JSON serializable data
        api_key: API key for authentication
        params: Optional query parameters
        data: Optional form data or plain text payload
        files: Optional files to upload
        timeout: Request timeout in seconds (default: 4 minutes)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
                     This will be exponentially increased with each retry
        retry_status_codes: List of HTTP status codes to retry (default: [502, 503, 504])

    Returns:
        dict: JSON response from the server

    Raises:
        RequestError: If API key is missing or request fails
    """
    if not api_key:
        raise RequestError("API key is required but not provided")

    # Initialize parameters
    headers = {"Authorization": f"Bearer {api_key}"}
    retry_status_codes = retry_status_codes or [502, 503, 504]

    # Track attempts
    attempt = 0
    last_exception = None

    while attempt <= max_retries:
        attempt += 1

        # Log the outgoing request with attempt number if retrying
        if attempt == 1:
            logger.info(f"Sending request to {url}")
        else:
            logger.info(f"Retry attempt {attempt - 1}/{max_retries} for request to {url}")

        try:
            # Make the request
            with httpx.Client(timeout=timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                )

            # Log the response
            logger.info(f"Received response from {url}: Status {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response content length: {len(response.content)}")

            # Check if we got a retriable status code
            if response.status_code in retry_status_codes and attempt <= max_retries:
                # Synchronous sleep for retry
                retry_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f"Received status {response.status_code} from {url}, "
                    f"retrying in {retry_time:.2f} seconds (attempt {attempt}/{max_retries})"
                )
                time.sleep(retry_time)
                continue

            # Raise exception for other error status codes
            response.raise_for_status()

            return response.json()

        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code in retry_status_codes and attempt <= max_retries:
                # Synchronous sleep for retry
                retry_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f"HTTP error {e.response.status_code} from {url}, "
                    f"retrying in {retry_time:.2f} seconds (attempt {attempt}/{max_retries})"
                )
                time.sleep(retry_time)
                continue
            else:
                # Non-retriable error or max retries exceeded
                if attempt > max_retries:
                    logger.error(f"HTTP error after {max_retries} retries: {str(e)}")
                raise RequestError.from_http_error(
                    e, context=f"Error making {method} request to {url}"
                ) from None

        except httpx.RequestError as e:
            last_exception = e
            if attempt <= max_retries:
                # Synchronous sleep for retry
                retry_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f"Network error: {str(e)} from {url}, "
                    f"retrying in {retry_time:.2f} seconds (attempt {attempt}/{max_retries})"
                )
                time.sleep(retry_time)
                continue
            else:
                error = RequestError(f"Network error: {e!s}")
                logger.error(f"Network error after {max_retries} retries: {str(error)}")
                raise error from None

        except Exception as e:
            last_exception = e
            error = RequestError(f"Unexpected error: {e!s}")
            logger.error(f"Unexpected error during request: {str(error)}")
            raise error from None

    # If we've exhausted all retries
    if last_exception:
        if isinstance(last_exception, httpx.HTTPStatusError):
            raise RequestError.from_http_error(
                last_exception, context=f"Request failed after {max_retries} retries"
            ) from None
        else:
            raise RequestError(
                f"Request failed after {max_retries} retries: {str(last_exception)}"
            ) from None

    # This should never happen, but just in case
    raise RequestError(f"Request failed after {max_retries} retries with unknown error")
