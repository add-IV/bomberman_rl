from os import path
import numpy as np
import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .mcts import MCTS
from .game_state import state_from_game_state
from .deep_network import MCTSNetwork, load_model

device = torch_directml.device()
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


def setup(self):
    """Setup code, called once at the start of the game"""
    self.logger.info("Setup called")
    self.model = load_model("mcts_model.pt")
    self.mcts = MCTS(
        self.model,
        time_limit=0.4,
        temperature=0,
        exploration_constant=1.0,
    )


def act(self, game_state: dict):
    """Act method returns the next action the agent will take"""
    action = self.mcts.search(game_state)
    return action
