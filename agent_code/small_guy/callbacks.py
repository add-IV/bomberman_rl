import torch
import numpy as np
from agent_code.small_guy.actions import ACTIONS
from agent_code.small_guy.game_state import state_from_game_state, GameState
from agent_code.small_guy.deep_network import load_model

def setup(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not self.train:
        self.model = load_model("small_guy.pt")
        self.epsilon = 0


def act(self, game_state: dict) -> str:
    network_input = state_from_game_state(GameState(**game_state))
    network_input = torch.tensor(network_input, dtype=torch.float32).to(self.device)
    
    if np.random.rand() > self.epsilon:
        with torch.no_grad():
            value, policy = self.model(network_input)
            action_index = policy.argmax().item()
    else:
        action_index = np.random.choice(range(6))

    return ACTIONS[action_index]