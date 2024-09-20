import time
import numpy as np

from agent_code.mcts.game_state import state_from_game_state
from agent_code.mcts.state_explorer import StateExplorer

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

    def __str__(self):
        return f"TreeNode(visits={self.visits}, total_value={self.total_value}, prior={self.prior})"



class MCTS:
    def __init__(
        self,
        policy_net,
        time_limit=0.1,
        exploration_constant=1 / np.sqrt(2),
        temperature=5,
    ):
        self.policy_net = policy_net
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.state_explorer = StateExplorer()
        self.root = None

    def search(self, game_state):
        root = TreeNode(None, None, game_state)
        start_time = time.time()
        i = 0
        while time.time() - start_time < self.time_limit:
            i += 1
            node = root
            while node.children:
                action = max(node.children, key=lambda a: self.ucb(node, a))
                node = node.children[action]
            if self.state_explorer.is_terminal(node.state):
                if self.state_explorer.is_dead(node.state):
                    value = -1
                else:
                    value=node.reward
            elif node.visits == 0:
                value = self.get_value(node.state) + node.reward
            else:
                value = self.expand(node)
            self.backup(node, value)
        action = max(root.children, key=lambda a: root.children[a].visits)
        print(i)
        print([f"{key}: {value.total_value / value.visits}" for key, value in root.children.items()])
        return action

    def ucb(self, parent, action):
        child = parent.children[action]
        return child.total_value / (child.visits if child.visits !=0 else 0.001) + self.exploration_constant * np.sqrt(
            np.log(parent.visits) / child.visits
        )

    def get_value(self, game_state):
        # use the policy network to get the value of the game state
        value = self.policy_net(state_from_game_state(game_state))[0]
        return value

    def expand(self, node):
        game_state = node.state
        current_reward = node.reward
        value, policy = self.policy_net(state_from_game_state(game_state))
        for action in ACTIONS:
            if self.state_explorer.is_action_valid(game_state, action):
                new_state, reward = self.state_explorer.step(game_state, action)
                if self.state_explorer.is_terminal(new_state):
                    if self.state_explorer.is_dead(new_state):
                        reward = -1
                    else:
                        reward=current_reward + reward
                node.children[action] = TreeNode(
                    node, action, new_state, prior=policy[0][ACTIONS.index(action)], reward=reward
                )
        return value

    def backup(self, node, value):
        while node:
            node.visits += 1
            node.total_value += value
            node = node.parent
