from __future__ import annotations

import logging
import socket
import time

from .main import SERVER_PORT

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds


def kill() -> None:
    """Kill the running emulator server.

    Connects to the emulator server and sends a kill command.
    Will retry connection up to MAX_RETRIES times.

    Raises:
        ConnectionError: If unable to connect after MAX_RETRIES attempts
        RuntimeError: If kill command fails
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
                logger.info(
                    "Waiting for server to start...",
                    extra={"attempt": attempt + 1, "max_retries": MAX_RETRIES},
                )
                time.sleep(RETRY_DELAY)
                connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    else:
        raise ConnectionError("Failed to connect after %d attempts", MAX_RETRIES) from last_error

    try:
        # Send kill command
        connection_type = "kill\n"
        connection.sendall(connection_type.encode("utf-8"))
        logger.info("Kill command sent successfully")
    except Exception as e:
        raise RuntimeError("Failed to send kill command") from e
    finally:
        connection.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    kill()
