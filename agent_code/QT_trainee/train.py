import numpy as np
import random
import events as e
from .constants import ACTIONS
from .features import BombermanFeatureExtractor  # Import the feature extractor
import pickle

NUM_EPISODES = 10000
learning_rate = 0.001
discount_factor = 0.95
epsilon = 0.1

def setup_training(self):
    self.feature_extractor = BombermanFeatureExtractor(self)
    self.q_table = {}
    load_model(self)
    self.logger.info(f"Training setup complete with {NUM_EPISODES} episodes.")

def get_escape_direction(self, game_state):
    x, y = game_state['self'][3]  # Agent's current position
    safe_directions = []

    for direction, (dx, dy) in zip(['UP', 'DOWN', 'LEFT', 'RIGHT'], [(0, -1), (0, 1), (-1, 0), (1, 0)]):
        new_position = (x + dx, y + dy)
        if is_safe_from_bombs(game_state, new_position):
            safe_directions.append(direction)

    return random.choice(safe_directions) if safe_directions else None


def is_safe_from_bombs(game_state, position):
    x, y = position

    for bomb in game_state['bombs']:
        # Ensure bomb data is correctly unpacked
        if isinstance(bomb, (tuple, list)) and len(bomb) in [2, 3]:
            bomb_x, bomb_y = bomb[0], bomb[1]
            timer = bomb[2] if len(bomb) == 3 else 3  # Default timer if not provided

            # Check if bomb_x and bomb_y are not tuples
            if isinstance(bomb_x, (tuple, list)):
                continue  # Skip if bomb_x is malformed
            if isinstance(bomb_y, (tuple, list)):
                continue  # Skip if bomb_y is malformed

            # Convert to floats
            bomb_x = float(bomb_x)
            bomb_y = float(bomb_y)
            x = float(x)
            y = float(y)

            # Check if the position is within the bomb's blast radius
            if (bomb_x == x and abs(bomb_y - y) <= timer) or (bomb_y == y and abs(bomb_x - x) <= timer):
                return False

    return True


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    if old_game_state is None or new_game_state is None:
        return

    old_state_encoded = self.feature_extractor.encode_state(old_game_state)
    new_state_encoded = self.feature_extractor.encode_state(new_game_state)
    old_q_value = self.q_table.get((old_state_encoded, self_action), 0)
    reward = reward_from_events(events) or 0

    if e.BOMB_DROPPED in events:
        escape_direction = get_escape_direction(self, new_game_state)
        if escape_direction:
            self_action = escape_direction

    future_q_values = [self.q_table.get((new_state_encoded, action), 0) for action in ACTIONS]
    max_future_q_value = max(future_q_values)
    delta = reward + discount_factor * max_future_q_value - old_q_value
    self.q_table[(old_state_encoded, self_action)] = old_q_value + learning_rate * delta

    self.logger.debug(f"Updated Q-value for state {old_state_encoded} and action {self_action}: {self.q_table[(old_state_encoded, self_action)]}")

def save_model(self):
    try:
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
        self.logger.info("Model saved to q_table.pkl")
    except Exception as ex:
        self.logger.error(f"Error saving model: {ex}")

def load_model(self):
    try:
        with open('q_table.pkl', 'rb') as f:
            self.q_table = pickle.load(f)
        self.logger.info("Q-table loaded from q_table.pkl")
    except FileNotFoundError:
        self.logger.warning("No model file found, starting with an empty Q-table.")
    except Exception as ex:
        self.logger.error(f"Error loading model: {ex}")

def reward_from_events(events: list):
    game_rewards = {
        e.COIN_COLLECTED: 10.0,
        e.KILLED_OPPONENT: 50.0,
        e.BOMB_DROPPED: -1.0,
        e.MOVED_DOWN: 0.1,
        e.MOVED_UP: 0.1,
        e.MOVED_RIGHT: 0.1,
        e.MOVED_LEFT: 0.1,
        e.INVALID_ACTION: -0.5,
        e.WAITED: -0.3,
        e.CRATE_DESTROYED: 5.0,
        e.GOT_KILLED: -100.0,
        e.KILLED_SELF: -500.0,
        e.SURVIVED_ROUND: 30.0
    }
    return sum(game_rewards.get(event, 0) for event in events)

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    self.logger.info("End of round.")
    if last_game_state is not None:
        game_events_occurred(self, last_game_state, last_action, None, events)
    save_model(self)
