from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any

from .main import SERVER_PORT

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds
BUFFER_SIZE = 1_000_000  # 1MB buffer for receiving data


def evaluate() -> dict[str, Any]:
    """Evaluate the current state of the emulator.

    Connects to the emulator server and requests evaluation results.
    Will retry connection up to MAX_RETRIES times.

    Returns:
        dict[str, Any]: Evaluation results from the emulator

    Raises:
        ConnectionError: If unable to connect after MAX_RETRIES attempts
        json.JSONDecodeError: If received data is not valid JSON
        RuntimeError: If evaluation request fails
    """
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    last_error = None

    # Try to connect with retries
    for attempt in range(MAX_RETRIES):
        try:
            connection.connect(("localhost", SERVER_PORT))
            break
        except ConnectionRefusedError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                logger.info("Waiting for server to start...")
                time.sleep(RETRY_DELAY)
                connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    else:
        raise ConnectionError(f"Failed to connect after {MAX_RETRIES} attempts") from last_error

    try:
        # Send evaluation request
        connection_type = "evaluate\n"
        connection.sendall(connection_type.encode("utf-8"))

        # Receive and parse result
        try:
            result = json.loads(connection.recv(BUFFER_SIZE).decode("utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError("Received invalid JSON from emulator") from e
        except Exception as e:
            raise RuntimeError("Failed to receive evaluation results") from e

        return result
    finally:
        connection.close()
