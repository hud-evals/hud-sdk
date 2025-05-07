from __future__ import annotations

import json
<<<<<<< HEAD
import socketserver
import sys
=======
import socket
>>>>>>> 92f0d50 (working local docker)
import time
from threading import Lock, Thread

from .emulator import Emulator

<<<<<<< HEAD
emulator: Emulator | None = None

lock = Lock()

class Server(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        global emulator

        if emulator is None:
            raise ValueError("Emulator not initialized")

        connection_type = self.rfile.readline().strip().decode("utf-8")
        print(f"Connection type: {connection_type}")
        if connection_type == "step":
            actions_raw = self.rfile.readline().strip().decode("utf-8")
            actions = json.loads(actions_raw)
            for action in actions:
                if action.get("type") == "press":
                    emulator.press_button_sequence(action.get("keys"))
            observation = emulator.get_observation()
            self.wfile.write(json.dumps(observation).encode("utf-8"))
        elif connection_type == "evaluate":
            evaluate_result = emulator.get_evaluate_result()
            self.wfile.write(json.dumps(evaluate_result).encode("utf-8"))
        elif connection_type == "kill":
            Thread(target=self.server.shutdown).start()
            self.server.server_close()
            return
        else:
            Thread(target=self.server.shutdown).start()
            self.server.server_close()
            raise ValueError(f"Unknown connection type: {connection_type}")


def process_signal_thread() -> None:
    with socketserver.TCPServer(("localhost", 6000), Server) as server:
        server.serve_forever()
    print("Server closed")


def main(game_name: str) -> None:
    global emulator
    emulator = Emulator(rom_path=f"gamefiles/{game_name}.gb")
    # print to stderr

=======
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
>>>>>>> 92f0d50 (working local docker)
    signal_thread = Thread(target=process_signal_thread)
    signal_thread.start()
    while True and signal_thread.is_alive():
        with lock:
            if not emulator.tick():
                break
        time.sleep(0.01)

    signal_thread.join()
    emulator.stop()
<<<<<<< HEAD


if __name__ == "__main__":
    main(sys.argv[1])
=======
>>>>>>> 92f0d50 (working local docker)
