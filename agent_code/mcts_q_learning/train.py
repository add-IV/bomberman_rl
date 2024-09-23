def setup_training(self):
    pass

def game_events_occurred(
    self,
    old_game_state: dict,
    last_action: str,
    new_game_state: dict,
    events: list[str],
):
    pass


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    self.logger.info(self.mcts)