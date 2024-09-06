"""This module contains the model for the DQN agent."""

import random
from collections import deque, namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_directml

device = torch_directml.device()

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
out_size = len(ACTIONS)
in_size = 15*15
in_size_after_conv = 11*11

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

class Coin_Finder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*in_size_after_conv, 32),
            nn.ReLU(),
            nn.Linear(32, out_size)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x[None, None, :, :]
        else:
            x = x[:, None, :, :]
        return super().forward(x)

    def select_action(self, state, eps_threshold=0.5, all_actions=False):
        n_actions = len(ACTIONS)

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                if all_actions:
                    return self(state)
                else:
                    return torch.argmax(self(state))
        else:
            return torch.tensor(
                [[random.randrange(n_actions)]], device=device, dtype=torch.long
            )