from os import path
import numpy as np
import torch
# import torch_directml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent_code.mcts.mcts import MCTS
from agent_code.mcts.game_state import state_from_game_state
from agent_code.mcts.deep_network import MCTSNetwork, load_model

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


def setup(self):
    """Setup code, called once at the start of the game"""
    self.logger.info("Setup called")
    curr_dir = path.dirname(path.abspath(__file__))
    self.model = load_model(path.join(curr_dir, "small_guy.pt"))
    self.mcts = MCTS(
        self.model,
        time_limit=0.4,
        temperature=0,
        exploration_constant=1.0,
        logger = self.logger
    )


def act(self, game_state: dict):
    """Act method returns the next action the agent will take"""
    with torch.no_grad():
        action = self.mcts.search(game_state)
    return action
