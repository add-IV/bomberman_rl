import numpy as np
import random
from agent_code.q_table.game_state import FeatureState, state_to_index
from agent_code.q_table.actions import ACTIONS
from agent_code.q_table.q_table import load_q_table

EPSILON = 0.3  # Exploration rate


def setup(self):
    """
    Setup the agent. This is where we initialize the Q-table and any other parameters.
    """
    self.epsilon = EPSILON if self.train else 0
    try:
        self.q_table = load_q_table()
    except FileNotFoundError:
        self.q_table = np.random.rand(FeatureState.feature_size(), len(ACTIONS))


def act(self, game_state: dict):
    """
    Choose an action based on the current game state using an epsilon-greedy strategy.
    """
    state_index = state_to_index(game_state)

    if random.uniform(0, 1) < self.epsilon:
        action = np.random.choice(ACTIONS)
        self.logger.debug("Exploring: chose action {}".format(action))
    else:
        action = ACTIONS[np.argmax(self.q_table[state_index])]
        self.logger.debug("Exploiting: chose action {}".format(action))

    return action
