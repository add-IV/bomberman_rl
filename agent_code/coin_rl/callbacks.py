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

from agent_code.coin_rl.model import *

device = torch_directml.device()
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def setup(self):
    """Setup code, called once at the start of the game"""
    self.logger.info("Setup called")
    if not self.train:
        self.model = DQN(PARAMETER_SIZE, len(ACTIONS)).to(device)
        if path.isfile("model.pth"):
            state_dict = torch.load("model.pth")
            self.model.load_state_dict(state_dict)


def act(self, game_state: dict):
    """Act method returns the next action the agent will take"""
    state = state_from_game_state(game_state)
    if self.train:
        action = self.policy_net.select_action(state)
    else:
        action = self.model.select_action(state, eps_threshold=0.0)
    return ACTIONS[action]