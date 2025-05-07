from __future__ import annotations

import json
import socket
import time
from threading import Lock, Thread

from .emulator import Emulator

emulator = Emulator(rom_path="gamefiles/Pokemon Red.gb")

lock = Lock()


def process_signal_thread() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 6000))
    server.listen()
    while True:
        client, addr = server.accept()
        with lock:
            actions = json.loads(client.recv(1000000).decode("utf-8"))
            for action in actions:
                if action.get("type") == "press":
                    emulator.press_button_sequence(action.get("keys"))
                elif action.get("type") == "wait":
                    frame_count = action.get("time") * 60 / 1000
                    for _ in range(int(frame_count)):
                        emulator.tick()

            observation = emulator.get_observation()
            client.sendall(json.dumps(observation).encode("utf-8"))
        client.close()


if __name__ == "__main__":
    signal_thread = Thread(target=process_signal_thread)
    signal_thread.start()
    while True and signal_thread.is_alive():
        with lock:
            if not emulator.tick():
                break
        time.sleep(0.01)

    signal_thread.join()
    emulator.stop()
