import random
import pickle
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
    def __init__(self):
        self.memory = deque()

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

    def append_to_file(self, path):
        with open(path, "ab") as f:
            pickle.dump(self.memory, f)

    def load_from_file(self, path):
        with open(path, "rb") as f:
            while True:
                try:
                    self.memory.extend(pickle.load(f))
                except EOFError:
                    break
