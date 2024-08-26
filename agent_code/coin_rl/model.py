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

# code is based on the following tutorial:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS is the exploration rate of the agent
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.8
EPS = 0.1
TAU = 0.005
LR = 1e-4

PARAMETER_SIZE = 98

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def state_from_game_state(game_state):
    # get the relevant information from the game state
    round = game_state["round"]
    step = game_state["step"]
    board = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]
    self = game_state["self"]
    self_score = self[1]
    self_bombs = self[2]
    self_position = self[3]
    others = game_state["others"]
    others_positions = [o[3] for o in others]

    # local view
    local_size = 3
    x, y = self_position
    padded_board = np.pad(board, local_size, mode="constant", constant_values=-1)

    # -1 for walls, 0 for free space, 1 for crates
    local_view = padded_board[x : x + 2 * local_size + 1, y : y + 2 * local_size + 1]

    # local coins
    local_coins = np.zeros((2 * local_size + 1, 2 * local_size + 1))
    for coin in coins:
        coin_x, coin_y = coin
        if (
            x - local_size <= coin_x <= x + local_size
            and y - local_size <= coin_y <= y + local_size
        ):
            local_coins[coin_x - x + local_size, coin_y - y + local_size] = 1

    state = np.concatenate([local_view, local_coins], axis=None)
    state = torch.tensor(state, device=device, dtype=torch.float32).view(1, -1)

    return state


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


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def select_action(self, state, eps_threshold=EPS):
        n_actions = len(ACTIONS)

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                return self(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(n_actions)]], device=device, dtype=torch.long
            )


def optimize_model(memory, policy_net, target_net, optimizer):
    batch_size = min(len(memory), BATCH_SIZE)
    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    indexes = action_batch.view(-1, 1)
    state_action_values = policy_net(state_batch).gather(1, indexes)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1)[0].detach()
        )

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

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
    torch.save(policy_net.state_dict(), "model.pth")
