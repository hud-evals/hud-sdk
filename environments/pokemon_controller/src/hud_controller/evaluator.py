from __future__ import annotations

import json
import socket
import time
from typing import Any


def evaluate() -> dict[str, Any]:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # If cant connect, try 5 times
    for _ in range(5):
        try:
            connection.connect(("localhost", 6000))
            break
        except ConnectionRefusedError:
            print("Waiting for server to start...")
            time.sleep(1)
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    connection_type = "evaluate\n"
    connection.sendall(connection_type.encode("utf-8"))

    # Receive result
    result = json.loads(connection.recv(1000000).decode("utf-8"))
    connection.close()
    return result
