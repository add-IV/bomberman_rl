import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import events as e
from collections import deque, namedtuple
from .model import DQN, ACTIONS, state_from_game_state
from .callbacks import hidden_layers, learning_rate

# Define the transition named tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0  # Initial value of epsilon (for exploration)
TAU = 0.005

steps_done = 0

parameter_size = 98

device = torch.device('cpu')

# Set path to save model
MODEL_SAVE_PATH = "my-saved-model.pt"

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device='cpu', dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device='cpu')
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
                                         key
                                     ] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

    # save the model
    torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)


# DQN training environment (optimizer, replay memory, etc)
def setup_training(self):
        # Initialize policy and target networks
        self.policy_net = DQN(parameter_size, len(ACTIONS), hidden_layers).to(device)
        if os.path.isfile(MODEL_SAVE_PATH):
            state_dict = torch.load(MODEL_SAVE_PATH)
            self.policy_net.load_state_dict(state_dict)

        self.target_net = DQN(parameter_size, len(ACTIONS), hidden_layers).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize replay memory
        self.replay_memory = ReplayMemory(BUFFER_SIZE)

        # Set up optimizer with AdamW
        self.optimizer = optim.AdamW(self.policy_net.parameters(), learning_rate, amsgrad=True)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):

    if old_game_state is not None:

        if e.COIN_COLLECTED in events:
            self.coins_collected += 1
        if old_game_state:
            old_state = state_from_game_state(old_game_state)
            new_state = state_from_game_state(new_game_state)
            reward = torch.tensor([reward_from_events(events)], device=device)
            action = torch.tensor([ACTIONS.index(last_action)], device=device)
            self.replay_memory.push(old_state, action, new_state, reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    if last_game_state:
        old_state = state_from_game_state(last_game_state)
    new_state = None
    reward = torch.tensor([reward_from_events(events)], device=device)
    action = torch.tensor([ACTIONS.index(last_action)], device=device)
    self.replay_memory.push(old_state, action, new_state, reward)

    if self.coins_collected > 0:
        self.logger.info(f"Coins collected: {self.coins_collected}")
        self.coins_collected = 0
    else:
        self.logger.info("No coins collected")
    optimize_model(self.replay_memory, self.policy_net, self.target_net, self.optimizer)


def reward_from_events(self, events: list) -> int:
    """
    Define a reward structure based on the game events.
    You can customize the rewards to encourage or discourage specific behaviors.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 3,
        e.GOT_KILLED: -7,
        e.KILLED_SELF: -10,
        e.WAITED: -1,
        e.INVALID_ACTION: -4,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 2,
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(map(str, events))}")
    return reward_sum