from __future__ import annotations

import logging
import socket
import subprocess
import sys

logger = logging.getLogger(__name__)

# List of supported games
available_games: list[str] = ["pokemon_red"]


def setup(game_name: str) -> None:
    """Initialize and start the game emulator.

    Args:
        game_name: Name of the game to run (must be in available_games)

    Raises:
        ValueError: If game_name is not in available_games
        RuntimeError: If emulator fails to start
    """
    # Validate if game is available
    if game_name not in available_games:
        raise ValueError(
            "Game %s is not available. Available games: %s",
            game_name,
            available_games,
        )

    # If there is already an emulator running, kill it and run a new one
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        connection.connect(("localhost", 6000))
        connection_type = "kill\n"
        connection.sendall(connection_type.encode("utf-8"))
        connection.close()
    except ConnectionRefusedError:
        logger.debug("No existing emulator found")
    except Exception as e:
        logger.warning("Error while checking for existing emulator", extra={"error": str(e)})

    # Run a new emulator
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "hud_controller.main", game_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if process.poll() is not None:
            raise RuntimeError("Emulator process failed to start")
    except Exception as e:
        raise RuntimeError(f"Failed to start emulator: {e}") from e


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    setup(sys.argv[1])
