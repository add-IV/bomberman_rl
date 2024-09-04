import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# epsilon parameters
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.995

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layers, device='cpu'):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_observations, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.layers.append(nn.Linear(hidden_layers[-1], n_actions))

        self.device = torch.device(device)
        self.to(self.device)  # Move model to the correct device

        # Epsilon parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    # Epsilon-greedy policy
    def select_action(self, state, steps_done):
        # Compute the epsilon threshold with decay
        # eps_threshold = max(self.epsilon_min, self.epsilon * (self.epsilon_decay ** steps_done))
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        sample = random.random()

        if sample > eps_threshold:
            # Use policy to select the best action
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                return self(state).max(1).indices.view(1, 1)
        else:
            # Select a random action
            return random.randrange(self.n_actions)

def state_from_game_state(game_state, local_size=3):
    """
    Extracts a localized state representation from the game state.

    Parameters:
        game_state (dict): The game state from which to extract information.
        local_size (int): The size of the local view around the player (default is 3).

    Returns:
        torch.Tensor: The processed state as a tensor, ready for input into the neural network.
    """

    # Player's current position
    self_position = game_state["self"][3]
    x, y = self_position

    # Extract relevant parts of the game state
    board = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]

    # Pad the board for out-of-bound areas when looking at the local view
    padded_board = np.pad(board, local_size, mode="constant", constant_values=-1)

    # Get the local view centered on the player
    local_view = padded_board[x: x + 2 * local_size + 1, y: y + 2 * local_size + 1]

    # Initialize matrix for local coins and bombs
    local_coins_and_bombs = np.zeros((2 * local_size + 1, 2 * local_size + 1))

    # Mark the positions of coins in the local view
    for coin_x, coin_y in coins:
        if abs(coin_x - x) <= local_size and abs(coin_y - y) <= local_size:
            local_coins_and_bombs[coin_x - x + local_size, coin_y - y + local_size] = 1

    # Mark the positions of bombs in the local view
    for bomb_pos, _ in bombs:
        bomb_x, bomb_y = bomb_pos
        if abs(bomb_x - x) <= local_size and abs(bomb_y - y) <= local_size:
            local_coins_and_bombs[bomb_x - x + local_size, bomb_y - y + local_size] = -1

    # Concatenate local view and coin/bomb information into a single array
    state = np.concatenate([local_view.flatten(), local_coins_and_bombs.flatten()])

    # Convert the state to a PyTorch tensor
    return torch.tensor(state, device='cpu', dtype=torch.float32).view(1, -1)



