from __future__ import annotations

import socket
import subprocess
import sys

from .kill import kill


def setup(game_name: str) -> None:
    # If there is already a  emulator running, kill it and run a new one
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        connection.connect(("localhost", 6000))
        connection_type = "kill\n"
        connection.sendall(connection_type.encode("utf-8"))
    except ConnectionRefusedError:
        pass
    # Run a new emulator
    subprocess.Popen(
        ["python", "-m", "hud_controller.main", game_name],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )


if __name__ == "__main__":
    setup(sys.argv[1])
