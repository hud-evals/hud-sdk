from __future__ import annotations

import json
import logging
import socketserver
import sys
import time
from threading import Lock, Thread

from .emulator import Emulator

# Constants
SERVER_PORT = 6000
TICK_SLEEP_TIME = 0.01  # seconds
FRAMES_PER_SECOND = 60

logger = logging.getLogger(__name__)

emulator: Emulator | None = None
lock = Lock()


class Server(socketserver.StreamRequestHandler):
    """TCP server handler for emulator control."""

    def handle(self) -> None:
        """Handle incoming connections and process commands.

        Raises:
            ValueError: If emulator is not initialized or unknown connection type
            json.JSONDecodeError: If invalid JSON is received
        """
        global emulator

        if emulator is None:
            raise ValueError("Emulator not initialized")

        try:
            connection_type = self.rfile.readline().strip().decode("utf-8")
            if connection_type == "step":
                self._handle_step()
            elif connection_type == "evaluate":
                self._handle_evaluate()
            elif connection_type == "kill":
                self._handle_kill()
            else:
                raise ValueError(f"Unknown connection type: {connection_type}")
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON received", extra={"error": str(e)})
            raise
        except Exception as e:
            logger.error("Error handling request", extra={"error": str(e)})
            raise

    def _handle_step(self) -> None:
        """Handle step command by processing actions and returning observation."""
        if emulator is None:
            raise ValueError("Emulator not initialized")

        actions_raw = self.rfile.readline().strip().decode("utf-8")
        actions = json.loads(actions_raw)
        with lock:
            for action in actions:
                if action.get("type") == "press":
                    emulator.press_button_sequence(action.get("keys"))
                elif action.get("type") == "wait":
                    frames = int(action.get("time") * FRAMES_PER_SECOND / 1000)
                    emulator.tick(frames)
            observation = emulator.get_observation()
        self.wfile.write(json.dumps(observation).encode("utf-8"))

    def _handle_evaluate(self) -> None:
        """Handle evaluate command by returning evaluation results."""
        if emulator is None:
            raise ValueError("Emulator not initialized")

        with lock:
            evaluate_result = emulator.get_evaluate_result()
        self.wfile.write(json.dumps(evaluate_result).encode("utf-8"))

    def _handle_kill(self) -> None:
        """Handle kill command by shutting down the server."""
        Thread(target=self.server.shutdown).start()
        self.server.server_close()


def process_signal_thread() -> None:
    """Run the TCP server in a separate thread."""
    try:
        with socketserver.TCPServer(("localhost", SERVER_PORT), Server) as server:
            logger.info("Server started", extra={"port": SERVER_PORT})
            server.serve_forever()
    except Exception as e:
        logger.error("Server error", extra={"error": str(e)})
    finally:
        logger.info("Server closed")


def main(game_name: str) -> None:
    """Main game loop.

    Args:
        game_name: Name of the game to run

    Raises:
        RuntimeError: If emulator fails to initialize
    """
    global emulator

    try:
        emulator = Emulator(rom_path=f"gamefiles/{game_name}.gb")
        logger.info("Emulator initialized", extra={"game": game_name})

        signal_thread = Thread(target=process_signal_thread)
        signal_thread.start()

        while signal_thread.is_alive():
            with lock:
                if not emulator.tick():
                    break
            time.sleep(TICK_SLEEP_TIME)

        signal_thread.join()
    except Exception as e:
        logger.error("Error in main loop", extra={"error": str(e)})
        raise RuntimeError(f"Emulator failed: {e}") from e
    finally:
        if emulator:
            emulator.stop()
            logger.info("Emulator stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main(sys.argv[1])
