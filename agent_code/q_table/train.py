
from agent_code.q_table.game_state import TargetKind, state_to_features, state_to_index
from agent_code.q_table.actions import ACTIONS
from agent_code.q_table.game_state import FeatureState
from agent_code.q_table.q_table import save_q_table
import events as e
from dataclasses import dataclass
import numpy as np

e.WAITED_WITHOUT_REASON = "WAITED_WITHOUT_REASON"

@dataclass
class TrainingDatapoint:
    index: int
    action: int
    reward: int
    next_index: int


def setup_training(self):
    self.k = FeatureState.feature_size()
    self.t = 10000
    self.q_table_copy = self.q_table.copy()
    self.training_data = []

def game_events_occurred(self, old_game_state: dict, self_action: str, _: dict, events: list[str]):
    index = state_to_index(old_game_state)
    action = ACTIONS.index(self_action)
    custom_events = get_custom_events(self_action, state_to_features(old_game_state))
    events.extend(custom_events)
    reward = reward_from_events(events)
    next_index = state_to_index(old_game_state)

    self.training_data.append(TrainingDatapoint(index, action, reward, next_index))


def get_custom_events(self_action: str, feature_state: FeatureState) -> list[str]:
    events = []
    if self_action == "Wait" and np.all(feature_state.danger == False):
        events.append(e.WAITED_WITHOUT_REASON)

    return events


def reward_from_events(events: list[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -1,
        e.GOT_KILLED: -4,
        e.INVALID_ACTION: -0.1,
        e.CRATE_DESTROYED: 0.1,
        e.COIN_FOUND: 1,
        e.SURVIVED_ROUND: 2,
        e.BOMB_DROPPED: 0.2,
        e.WAITED_WITHOUT_REASON: -0.1
    }

    reward = sum(game_rewards.get(event, 0) for event in events)
    return reward

def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    index = state_to_index(last_game_state)
    action = ACTIONS.index(last_action)
    reward = reward_from_events(events)
    next_index = index

    self.training_data.append(TrainingDatapoint(index, action, reward, next_index))

    train(self)
    save_q_table(self)
    self.q_table_copy = self.q_table.copy()

def train(self):
    for data in self.training_data:
        self.t += 1
        old_q_value = self.q_table_copy[data.index, data.action]
        learning_rate = self.k / (self.k + self.t)
        new_q_value = old_q_value + learning_rate * (data.reward + max(self.q_table_copy[data.next_index]) - old_q_value)
        self.q_table[data.index, data.action] = new_q_value
    
    self.training_data = []