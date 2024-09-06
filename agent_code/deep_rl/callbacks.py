from os import path
import numpy as np
import json
import math
import random

import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent_code.deep_rl.model import *
from agent_code.deep_rl.deep_rl import state_from_game_state

device = torch_directml.device()
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def setup(self):
    """Setup code, called once at the start of the game"""
    self.logger.info("Setup called")
    if not self.train:
        self.model = Coin_Finder().to(device)
        if path.isfile("model.pth"):
            state_dict = torch.load("model.pth")
            self.model.load_state_dict(state_dict)


def act(self, game_state: dict):
    """Act method returns the next action the agent will take"""
    state = state_from_game_state(game_state)
    if self.train:
        action = self.policy_net.select_action(state)
    else:
        actions = self.model.select_action(state, eps_threshold=0.0, all_actions=True).cpu().numpy()
        sorted_actions = np.argsort(actions)[0][::-1]
        for action in sorted_actions:
            if is_action_valid(game_state, ACTIONS[action]):
                return ACTIONS[action]
    return ACTIONS[action]


def is_action_valid(game_state, action):
    """Check if an action is valid"""
    x, y = game_state["self"][3]
    if action == "UP":
        return game_state["field"][x, y - 1] == 0
    elif action == "DOWN":
        return game_state["field"][x, y + 1] == 0
    elif action == "LEFT":
        return game_state["field"][x - 1, y] == 0
    elif action == "RIGHT":
        return game_state["field"][x + 1, y] == 0
    else:
        return True