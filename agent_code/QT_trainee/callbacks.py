import numpy as np
import random

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']  # Possible actions
EPSILON = 0.1  # Exploration rate


def setup(self):
    """
    Setup the agent. This is where we initialize the Q-table and any other parameters.
    """
    self.q_table = np.zeros((10000, len(ACTIONS)))  # Q-table initialization (states x actions)
    self.epsilon = EPSILON  # Exploration-exploitation tradeoff
    self.logger.info("Q-table initialized with shape: {}".format(self.q_table.shape))


def act(self, game_state: dict):
    """
    Choose an action based on the current game state using an epsilon-greedy strategy.
    """
    state_index = state_to_index(game_state, self.q_table)

    # Epsilon-greedy action selection
    if random.uniform(0, 1) < self.epsilon:  # Explore: choose a random action
        action = np.random.choice(ACTIONS)
        self.logger.debug("Exploring: chose action {}".format(action))
    else:  # Exploit: choose the best action based on the Q-table
        action = ACTIONS[np.argmax(self.q_table[state_index])]
        self.logger.debug("Exploiting: chose action {}".format(action))

    return action


def state_to_index(game_state, q_table):
    """
    Converts the current game state into an index for the Q-table.
    This allows us to map states to rows in the Q-table.
    """
    if game_state is None:
        return 0  # Return a default state index if game_state is None

    # Example state: based on player's position, number of coins, and bombs
    player_position = game_state.get('self', (None, None, None, (0, 0)))[3]
    coin_positions = game_state.get('coins', [])
    bombs = game_state.get('bombs', [])

    # Create a simplified representation of the game state
    state = (player_position, len(coin_positions), len(bombs))

    # Hash the state to an index
    return hash(state) % len(q_table)
