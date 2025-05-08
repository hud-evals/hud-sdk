from __future__ import annotations

import socket
import time


def kill() -> None:
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

    connection_type = "kill\n"
    connection.sendall(connection_type.encode("utf-8"))


if __name__ == "__main__":
    kill()
