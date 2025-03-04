from typing import Any

class Agent:
    def __init__(self, client: Any):
        self.client = client
        self.messages = []
        self.responses = []

    def predict(self):
        raise NotImplementedError("Subclasses must implement this method")
