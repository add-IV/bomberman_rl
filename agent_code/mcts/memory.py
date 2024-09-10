import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

    def append_to_file(self, path):
        with open(path, "a", encoding="utf-8") as f:
            for transition in self.memory:
                f.write(f"{transition}\n")
