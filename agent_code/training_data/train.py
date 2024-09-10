from .memory import ReplayMemory

def setup_training(self):
    self.replay_memory = ReplayMemory()


def game_events_occurred(
    self,
    old_game_state: dict,
    last_action: str,
    new_game_state: dict,
    events: list[str],
):
    self.replay_memory.push(old_game_state, last_action, new_game_state, events)
    

def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    self.replay_memory.push(last_game_state, last_action, None, events)

    if last_game_state["round"] % 100 == 0:
        self.replay_memory.append_to_file("training_data.pt")
        self.replay_memory.memory.clear()