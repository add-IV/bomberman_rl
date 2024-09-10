from collections import namedtuple
import numpy as np
from os import path

import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent_code.deep_rl.model import *
from agent_code.deep_rl.deep_rl import state_from_game_state, optimize_model
import events as e

device = torch_directml.device()

e.BACK_AND_FORTH = "back_and_forth"
e.TOWARDS_COIN_DENSITY = "towards_coin_density"
e.AWAY_FROM_COIN_DENSITY = "away_from_coin_density"


def reset_counters(self):
    self.coins_collected = 0
    self.total_reward = 0
    self.event_total = {
        event: 0
        for event in e.__dict__.values()
        if isinstance(event, str) and event.isupper()
    }
    
    self.action_memory = [-1] * 3
    self.replay_memory = ReplayMemory(600)


def setup_training(self):
    reset_counters(self)
    self.policy_net = Coin_Finder().to(device)
    if path.isfile("model.pth"):
        state_dict = torch.load("model.pth")
        self.policy_net.load_state_dict(state_dict)
    self.target_net = Coin_Finder().to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
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

    custom_events = get_custom_events(last_action, self.action_memory, old_game_state)
    events.extend(custom_events)
    old_state = state_from_game_state(old_game_state)
    new_state = state_from_game_state(new_game_state)
    reward = torch.tensor([reward_from_events(events)], device=device)
    action = torch.tensor([ACTIONS.index(last_action)], device=device)
    self.replay_memory.push(old_state, action, new_state, reward)

    for event in events:
        self.event_total[event] += 1
    self.total_reward += reward.item()


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    if last_game_state:
        old_state = state_from_game_state(last_game_state)

    custom_events = get_custom_events(last_action, self.action_memory, last_game_state)
    events.extend(custom_events)

    reward = torch.tensor([reward_from_events(events)], device=device)
    action = torch.tensor([ACTIONS.index(last_action)], device=device)
    self.replay_memory.push(old_state, action, None, reward)

    for event in events:
        self.event_total[event] += 1
    self.total_reward += reward.item()

    if self.coins_collected > 0:
        self.logger.info(f"Coins collected: {self.coins_collected}")
    else:
        self.logger.info("No coins collected")
    self.logger.info(f"Total reward: {self.total_reward}")
    self.logger.info(f"Events: {self.event_total}")
    optimize_model(self.replay_memory, self.policy_net, self.target_net, self.optimizer)
    reset_counters(self)


def reward_from_events(events: list[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: -2,
        e.WAITED: -0.05,
        e.MOVED_DOWN: 0.01,
        e.MOVED_LEFT: 0.01,
        e.MOVED_RIGHT: 0.01,    
        e.MOVED_UP: 0.01,
        e.INVALID_ACTION: -0.1,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.SURVIVED_ROUND: 2,
        e.BOMB_EXPLODED: 0.2,
        e.BACK_AND_FORTH: -0.001,
        e.TOWARDS_COIN_DENSITY: 0.1,
        e.AWAY_FROM_COIN_DENSITY: -0.1,
    }

    reward = sum([game_rewards.get(event, 0) for event in events])
    return reward


def get_custom_events(last_action, action_memory, old_game_state):
    events = []
    # move back and forth
    if (
        action_memory[0] == action_memory[2]
        and action_memory[1] == last_action
        and action_memory[1] != last_action
    ):
        events.append(e.BACK_AND_FORTH)
    # move_towards_coin_density_if_no_local_coin
    coins = old_game_state["coins"]
    postition = old_game_state["self"][3]
    return events
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
