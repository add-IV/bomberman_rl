import numpy as np
import random
import events as e
from .callbacks import state_to_index, ACTIONS

# Hyperparameters
LEARNING_RATE = 0.1  # Alpha
DISCOUNT_FACTOR = 0.95  # Gamma
EPSILON = 0.1  # Epsilon for exploration

def setup_training(self):
    """
    Setup variables for training. This function is called before training starts.
    """
    self.learning_rate = LEARNING_RATE
    self.discount_factor = DISCOUNT_FACTOR
    self.epsilon = EPSILON
    self.logger.info("Training setup complete.")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    """
    Update the Q-table based on the events that occurred in the game. This is where learning happens.
    """
    if old_game_state is None or new_game_state is None:
        return

    # Convert the states into indices
    old_state_index = state_to_index(old_game_state, self.q_table)
    new_state_index = state_to_index(new_game_state, self.q_table)

    # Get the action index (which corresponds to the action taken)
    action_index = ACTIONS.index(self_action)

    # Calculate the reward from the events
    reward = reward_from_events(self, events)

    # Q-learning update: old value + alpha * (reward + gamma * max_future_value - old_value)
    old_value = self.q_table[old_state_index, action_index]
    future_value = np.max(self.q_table[new_state_index])  # Max Q-value for the next state

    # Q-learning update rule
    self.q_table[old_state_index, action_index] = old_value + self.learning_rate * (
        reward + self.discount_factor * future_value - old_value
    )

    self.logger.debug(f"Updated Q-table at state {old_state_index} for action {self_action}")


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    This function is called at the end of each game round. It's a good place to save the Q-table.
    """
    self.logger.info("End of round.")
    game_events_occurred(self, last_game_state, last_action, None, events)


def reward_from_events(self, events: list):
    """
    Calculate a reward based on the events that occurred during this game step.
    """
    # Simple reward structure
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 10
    if e.KILLED_OPPONENT in events:
        reward += 50
    if e.GOT_KILLED in events:
        reward -= 100
    if e.CRATE_DESTROYED in events:
        reward += 5

    return reward
