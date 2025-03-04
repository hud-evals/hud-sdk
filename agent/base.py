class Agent:
    def __init__(self):
        self.messages = []
        self.responses = []

    def predict(self):
        raise NotImplementedError("Subclasses must implement this method")
