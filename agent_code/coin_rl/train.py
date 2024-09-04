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

e.BACK_AND_FORTH = "back_and_forth"
e.TOWARDS_COIN_DENSITY = "towards_coin_density"
e.AWAY_FROM_COIN_DENSITY = "away_from_coin_density"

device = torch_directml.device()

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def setup_training(self):
    self.coins_collected = 0
    self.action_memory = [-1] * 3
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
        custom_events = get_custom_events(last_action, self.action_memory, old_game_state)
        events.extend(custom_events)
        old_state = state_from_game_state(old_game_state)
        new_state = state_from_game_state(new_game_state)
        reward = torch.tensor([reward_from_events(events, self.coins_collected)], device=device)
        action = torch.tensor([ACTIONS.index(last_action)], device=device)
        self.replay_memory.push(old_state, action, new_state, reward)


def get_custom_events(last_action, action_memory, old_game_state):
    events = []
    # move back and forth
    if action_memory[0] == action_memory[2] and action_memory[1] == last_action and action_memory[1] != last_action:
        events.append(e.BACK_AND_FORTH)
    # move_towards_coin_density_if_no_local_coin
    coins = old_game_state["coins"]
    postition = old_game_state["self"][3]
    if local_coins_available(coins, postition):
        return events
    c_metric = coin_metric(coins, postition)
    direction = np.argmax(c_metric)
    action = ACTIONS[direction]
    if action == last_action:
        events.append(e.TOWARDS_COIN_DENSITY)
    else:
        events.append(e.AWAY_FROM_COIN_DENSITY)
    return events

def local_coins_available(coins, position):
    for coin in coins:
        coin_x, coin_y = coin
        if abs(coin_x - position[0]) <= 3 and abs(coin_y - position[1]) <= 3:
            return True
    return False


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    if last_game_state:
        old_state = state_from_game_state(last_game_state)
    new_state = None
    custom_events = get_custom_events(last_action, self.action_memory, last_game_state)
    events.extend(custom_events)
    reward = torch.tensor([reward_from_events(events, self.coins_collected)], device=device)
    action = torch.tensor([ACTIONS.index(last_action)], device=device)
    self.replay_memory.push(old_state, action, new_state, reward)

    if self.coins_collected > 0:
        self.logger.info(f"Coins collected: {self.coins_collected}")
        self.coins_collected = 0
    else:
        self.logger.info("No coins collected")
    self.action_memory = [-1] * 3
    optimize_model(self.replay_memory, self.policy_net, self.target_net, self.optimizer)


def reward_from_events(events: list[str], coins_amount) -> int:
    coin_find_reward = (coins_amount * 1 / 30) ** 2 + 0.5
    game_rewards = {
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -5,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -0.1,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: coin_find_reward,
        e.BACK_AND_FORTH: -0.001,
        e.TOWARDS_COIN_DENSITY: 0.1,
        e.AWAY_FROM_COIN_DENSITY: -0.1,
    }

    reward = sum([game_rewards.get(event, 0) for event in events])
    return reward
