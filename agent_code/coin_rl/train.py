from collections import namedtuple, deque
import numpy as np
import json
import math
import random
from os import path

import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent_code.coin_rl.model import *
import events as e

device = torch_directml.device()

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def setup_training(self):
    self.coins_collected = 0
    self.policy_net = DQN(PARAMETER_SIZE, len(ACTIONS)).to(device)
    if path.isfile("model.pth"):
        state_dict = torch.load("model.pth")
        self.policy_net.load_state_dict(state_dict)
    self.target_net = DQN(PARAMETER_SIZE, len(ACTIONS)).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.replay_memory = ReplayMemory(600)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

def game_events_occurred(
    self,
    old_game_state: dict,
    last_action: str,
    new_game_state: dict,
    events: list[str],
):
    if e.COIN_COLLECTED in events:
        self.coins_collected += 1
    if old_game_state:
        old_state = state_from_game_state(old_game_state)
        new_state = state_from_game_state(new_game_state)
        reward = torch.tensor([reward_from_events(events)], device=device)
        action = torch.tensor([ACTIONS.index(last_action)], device=device)
        self.replay_memory.push(old_state, action, new_state, reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    if last_game_state:
        old_state = state_from_game_state(last_game_state)
    new_state = None
    reward = torch.tensor([reward_from_events(events)], device=device)
    action = torch.tensor([ACTIONS.index(last_action)], device=device)
    self.replay_memory.push(old_state, action, new_state, reward)
    
    if self.coins_collected > 0:
        self.logger.info(f"Coins collected: {self.coins_collected}")
        self.coins_collected = 0
    optimize_model(self.replay_memory, self.policy_net, self.target_net, self.optimizer)

def reward_from_events(events: list[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -5,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -0.1,
        e.BOMB_DROPPED: -0.1,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.5,
    }

    reward = sum([game_rewards.get(event, 0) for event in events])
    return reward
