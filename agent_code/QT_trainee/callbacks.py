import os
import random
import pickle
import numpy as np
from .constants import ACTIONS
from .features import BombermanFeatureExtractor

EPSILON = 0.1  # Exploration rate

def setup(self):
    """
    Setup the agent. Initialize the Q-table and any other parameters.
    """
    self.q_table = {}
    self.feature_extractor = BombermanFeatureExtractor(self)

    # Load Q-table if it exists
    model_filename = 'q_table.pkl'
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            self.q_table = pickle.load(f)
        self.logger.info(f"Loaded Q-table from {model_filename}")
    else:
        self.logger.info("Q-table initialized from scratch.")

    self.epsilon = EPSILON  # Exploration-exploitation tradeoff

def act(self, game_state: dict):
    """
    Choose an action based on the current game state using an epsilon-greedy strategy.
    """
    if game_state is None:
        return np.random.choice(ACTIONS)  # Random action if game state is None

    state = self.feature_extractor.encode_state(game_state)  # Encode the state

    # Epsilon-greedy action selection
    if random.uniform(0, 1) < self.epsilon:  # Explore
        action = np.random.choice(ACTIONS)
        self.logger.debug(f"Exploring: chose action {action}")
    else:  # Exploit
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in ACTIONS}  # Initialize Q-values for the state
        action = max(self.q_table[state], key=self.q_table[state].get)
        self.logger.debug(f"Exploiting: chose action {action}")

    return action
