from agent_code.mcts_q_learning.mcts import MCTS
from agent_code.mcts_q_learning.q_table import load_q_table


def setup(self):
    """Setup code, called once at the start of the game"""
    self.logger.info("Setup called")
    self.model = load_q_table()
    self.mcts = MCTS(
        self.model,
        time_limit=0.4,
        temperature=0,
        exploration_constant=1.0,
        logger=self.logger,
    )


def act(self, game_state: dict):
    """Act method returns the next action the agent will take"""
    action = self.mcts.search(game_state)
    return action
