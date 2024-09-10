from os import path
import numpy as np
import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .mcts import MCTS, state_from_game_state
from .deep_network import MCTSNetwork, load_model

device = torch_directml.device()
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


def setup(self):
    """Setup code, called once at the start of the game"""
    self.logger.info("Setup called")
    if not self.train:
        self.model = MCTSNetwork()
        self.mcts = MCTS(
            self.model,
            time_limit=0.4,
            max_score=50,
            temperature=0,
            exploration_constant=1.0,
        )


def act(self, game_state: dict):
    """Act method returns the next action the agent will take"""
    state = state_from_game_state(game_state)
    action = self.mcts.search(state)
    return action
