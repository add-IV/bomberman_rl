import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_directml

device = torch_directml.device()


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


class TreeNode:
    def __init__(self, parent, action, game_state, prior=0):
        self.parent = parent
        self.action = action
        self.state = game_state
        self.children = {}  # action -> TreeNode
        self.visits = 0
        self.total_value = 0
        self.prior = prior

    def __str__(self):
        return f"TreeNode(visits={self.visits}, total_value={self.total_value}, prior={self.prior})"


class StateExplorer:
    def __init__(self):
        pass

    def is_terminal(self, game_state):
        # either the player is dead or the game is over
        # game is over if no coins, crates, or players are left
        self_dead = game_state["self"][0] == 0
        others_dead = game_state["others"] == []
        coins = game_state["coins"] != []
        crates = 1 in game_state["field"]
        return self_dead or (others_dead and not coins and not crates)

    def get_valid_actions(self, game_state):
        return [
            action for action in ACTIONS if self.is_action_valid(game_state, action)
        ]

    def get_score(self, game_state):
        return game_state["self"][1]

    def step(self, game_state, action):
        if not self.is_action_valid(game_state, action):
            raise ValueError(f"Invalid action: {action}")
        new_game_state = game_state.copy()
        new_game_state["step"] += 1
        new_game_state["self"] = self.step_player(game_state["self"], action)
        new_game_state["others"] = [
            self.step_player(player, action) for player in game_state["others"]
        ]
        new_game_state["coins"] = self.step_coins(game_state["coins"], new_game_state)
        new_game_state["bombs"] = [self.step_bomb(bomb) for bomb in game_state["bombs"]]
        new_game_state["explosion_map"] = self.step_explosions(
            game_state["explosion_map"]
        )

        return new_game_state

    def is_action_valid(self, game_state, action):
        self_position = game_state["self"][3]
        x, y = self_position
        if action == "UP":
            return game_state["field"][x - 1, y] == 0
        elif action == "DOWN":
            return game_state["field"][x + 1, y] == 0
        elif action == "LEFT":
            return game_state["field"][x, y - 1] == 0
        elif action == "RIGHT":
            return game_state["field"][x, y + 1] == 0
        elif action == "WAIT":
            return True
        elif action == "BOMB":
            return game_state["self"][2] == True
        else:
            raise ValueError(f"Invalid action: {action}")


class MCTS:
    def __init__(
        self,
        policy_net,
        time_limit=0.1,
        exploration_constant=1 / np.sqrt(2),
        temperature=5,
        max_score=50,
    ):
        self.policy_net = policy_net
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.max_score = max_score
        self.state_explorer = StateExplorer()

    def search(self, game_state):
        root = TreeNode(None, None, game_state)
        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            node = root
            while node.children:
                action = max(node.children, key=lambda a: self.ucb(node, a))
                node = node.children[action]
            if self.state_explorer.is_terminal(node.state):
                value = self.state_explorer.get_final_score(node.state)
            elif node.visits == 0:
                value = self.get_value(node.state)
            else:
                value = self.expand(node)
            self.backup(node, value)
        return max(root.children, key=lambda a: root.children[a].visits)

    def ucb(self, parent, action):
        child = parent.children[action]
        return child.total_value / child.visits + self.exploration_constant * np.sqrt(
            np.log(parent.visits) / child.visits
        )

    def get_value(self, game_state):
        # use the policy network to get the value of the game state
        value = self.policy_net(game_state)[0]
        print(value)
        return value

    def expand(self, node):
        game_state = node.state
        policy, value = self.policy_net(game_state)
        for action in ACTIONS:
            if self.state_explorer.is_action_valid(game_state, action):
                new_state = self.state_explorer.step(game_state, action)
                node.children[action] = TreeNode(
                    node, action, new_state, prior=policy[ACTIONS.index(action)]
                )
        return value

    def backup(self, node, value):
        while node:
            node.visits += 1
            node.total_value += value
            node = node.parent
