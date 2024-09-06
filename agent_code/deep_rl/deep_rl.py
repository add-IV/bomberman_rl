from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_directml

from agent_code.deep_rl.model import *

device = torch_directml.device()

GAMMA = 0.8
EPS = 0.2
TAU = 0.005
LR = 1e-4


def state_from_game_state(game_state):
    # get the relevant information from the game state
    round = game_state["round"]
    step = game_state["step"]
    board = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    self = game_state["self"]
    self_score = self[1]
    self_bomb_available = self[2]
    self_position = self[3]
    others = game_state["others"]
    others_score = [o[1] for o in others]
    other_bomb_available = [o[2] for o in others]
    others_positions = [o[3] for o in others]

    # legend:
    # 0 = free space
    # 1 = wall
    # 2 = crate
    # 3 = coin
    # 4 = player

    # create the state tensor based on the board
    state = board[1:-1, 1:-1]
    state[state == 1] = 2
    state[state == -1] = 1

    # add the coins to the state tensor
    for coin in coins:
        x, y = coin
        state[x - 1, y - 1] = 3

    state[self_position[0] - 1, self_position[1] - 1] = 4

    # create the state tensor
    state = torch.tensor(state, device=device, dtype=torch.float)

    return state


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < 1:
        return
    batch_size = len(memory)
    transitions = memory.get()

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state).to(device)
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
