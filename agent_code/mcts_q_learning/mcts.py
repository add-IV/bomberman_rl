import time
import numpy as np

from agent_code.mcts_q_learning.game_state import state_to_index
from agent_code.mcts_q_learning.state_explorer import StateExplorer, GameState

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


class TreeNode:
    def __init__(self, parent, action, game_state, prior=0, reward=0):
        self.parent = parent
        self.action = action
        self.state = game_state
        self.children = {}  # action -> TreeNode
        self.visits = 0
        self.total_value = 0
        self.prior = prior
        self.reward = reward
        self.max_value = 0

    def __str__(self):
        return f"{self.action}: {self.total_value/(self.visits + 1)} [{self.visits}]"


class MCTS:
    def __init__(
        self,
        q_table,
        time_limit=0.1,
        exploration_constant=0.5 / np.sqrt(2),
        temperature=5,
        logger=None,
    ):
        self.q_table = q_table
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.state_explorer = StateExplorer()
        self.root = None
        self.logger = logger

    def search(self, game_state_dict):
        game_state = GameState(**game_state_dict)
        root = TreeNode(None, None, game_state)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = root
            while node.children:
                action = max(node.children, key=lambda a: self.ucb(node, a))
                node = node.children[action]
            if self.state_explorer.is_terminal(node.state):
                if self.state_explorer.is_dead(node.state):
                    value = -1
                else:
                    value = node.reward
            elif node.visits == 0:
                value = node.reward
            else:
                value = self.expand(node)
            self.backup(node, value)
        if not root.children:
            # accept defeat
            return "WAIT"
        action = max(root.children, key=lambda a: root.children[a].visits)
        print(
            [
                f"{key}: {value.total_value / value.visits} [{value.visits}]"
                for key, value in root.children.items()
            ]
        )
        self.log_tree(root)
        return action

    def ucb(self, parent, action):
        child = parent.children[action]
        return child.total_value / (
            child.visits if child.visits != 0 else 0.001
        ) + self.exploration_constant * np.sqrt(np.log(parent.visits) / child.visits)

    def use_policy_net(self, game_state):
        policy = self.q_table[state_to_index(game_state)]
        return policy

    def expand(self, node):
        game_state = node.state
        policy = self.use_policy_net(game_state)
        for action in ACTIONS:
            if self.state_explorer.is_action_valid(game_state, action):
                new_state, reward = self.state_explorer.step(game_state, action)
                if self.state_explorer.is_terminal(new_state):
                    if self.state_explorer.is_dead(new_state):
                        reward = -1
                node.children[action] = TreeNode(
                    node,
                    action,
                    new_state,
                    prior=policy[ACTIONS.index(action)],
                    reward=reward,
                )
        if not node.children:
            return -1
        return reward

    def backup(self, node, value):
        discounted_visits = 1
        while node:
            node.visits += 1
            node.total_value += value
            value *= 0.93
            discounted_visits *= 0.93
            node = node.parent

    def log_tree(self, node):

        def get_node_string(node, depth=0):
            node_string = f"{'  ' * depth}{node}"
            for child in node.children.values():
                node_string += f"\n{get_node_string(child, depth + 1)}"
            return node_string

        self.logger.info(get_node_string(node))
