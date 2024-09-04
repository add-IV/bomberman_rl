import os
import torch as T
import pickle
import numpy as np
import random
from .model import DQN, ACTIONS
from .state_to_features import state_to_features

PARAMETERS = 'final_parameters'  # Path to saved model parameters

learning_rate = 0.001
hidden_layers = [128, 128]

def setup(self):
    """
    Setup your agent. This is called once when loading each agent.
    """

    self.global_counter = 0

    # Load or initialize the model
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model")
        hidden_layers = [128, 128]
        self.model = DQN(3, len(ACTIONS), hidden_layers)  # Assuming input_dim = 4, and 6 possible actions
        self.optimizer = T.optim.Adam(self.model.parameters(), lr=0.001)  # Optimizer
        self.criterion = T.nn.MSELoss()  # Loss function
        self.epsilon = 1.0  # Initial exploration rate
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.epsilon = 0.1  # Lower epsilon during evaluation


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    """
    # Decay epsilon over time (Exploration vs Exploitation)
    self.global_counter += 1
    if self.train:
        if self.global_counter > 400 * 100:
            self.epsilon = max(0.3, self.epsilon * 0.99)
        if self.global_counter > 400 * 200:
            self.epsilon = max(0.1, self.epsilon * 0.99)
        if self.global_counter > 400 * 400:
            self.epsilon = max(0.0, self.epsilon * 0.99)

    # Exploration: Choose random action
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .1])

    # Exploitation: Use the model to predict the best action
    self.logger.debug("Querying model for action.")

    state = state_to_features(game_state)  # Convert the game state to a feature vector
    state = T.tensor(state, dtype=T.float32).unsqueeze(0)  # Convert to tensor and add batch dimension

    with T.no_grad():
        q_values = self.model(state)  # Get Q-values from the model
        action_index = T.argmax(q_values).item()  # Get the index of the best action

    return ACTIONS[action_index]



