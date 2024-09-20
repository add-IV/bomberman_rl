import numpy as np
import copy


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]

class StateExplorer:
    def __init__(self):
        pass

    def is_terminal(self, game_state):
        # either the player is dead or the game is over
        # game is over if no coins, crates, or players are left
        if game_state["self"][0] == "dead":
            return True
        others_dead = game_state["others"] == []
        coins = game_state["coins"] != []
        crates = 1 in game_state["field"]
        if others_dead and not coins and not crates:
            return True
        return False
    
    def is_dead(self, game_state):
        return game_state["self"][0] == "dead"


    def get_valid_actions(self, game_state):
        return [
            action for action in ACTIONS if self.is_action_valid(game_state, action)
        ]

    def get_score(self, game_state):
        return game_state["self"][1]

    def step(self, game_state, action):
        if not self.is_action_valid(game_state, action):
            raise ValueError(f"Invalid action: {action}")
        
        self.new_game_state = {}
        self.new_game_state["others"] = []
        self.new_game_state["coins"] = []
        self.new_game_state["bombs"]  = []
        self.new_game_state["field"] = copy.deepcopy(game_state["field"])
        self.new_game_state["step"] = game_state["step"] + 1

        self.step_player(action, game_state)
        self.step_others(game_state["others"])
        reward = self.step_coins(game_state["coins"])
        self.step_explosions(game_state["explosion_map"])
        reward += self.step_bombs(game_state["bombs"])
        reward += self.evaluate_explosions()

        return self.new_game_state, reward
    
    def step_player(self, action, game_state):
        self_position = game_state["self"][3]
        can_bomb = game_state["self"][2]
        x, y = self_position
        if action == "UP":
            self_position = (x, y - 1)
        elif action == "DOWN":
            self_position = (x, y + 1)
        elif action == "LEFT":
            self_position = (x - 1, y)
        elif action == "RIGHT":
            self_position = (x + 1, y)
        elif action == "WAIT":
            pass
        elif action == "BOMB":
            if can_bomb:
                self.new_game_state["bombs"].append([self_position, 4])
                can_bomb = False
        else:
            raise ValueError(f"Invalid action: {action}")
        self.new_game_state["self"] = (
            game_state["self"][0],
            game_state["self"][1],
            can_bomb,
            self_position,
        )
        
    def step_others(self, others):
        for other in others:
            self.new_game_state["others"].append(other)

    def step_coins(self, coins):
        reward = 0
        for coin in coins:
            collected = False
            if coin == self.new_game_state["self"][3]:
                reward += 0.2
                collected = True
            for other in self.new_game_state["others"]:
                if coin == other[3]:
                    collected = True
            if not collected:
                self.new_game_state["coins"].append(coin)                    
        return 0
    
    def step_explosions(self, explosion_map):
        self.new_game_state["explosion_map"] = np.clip(explosion_map - 1, -1, 2)

    def get_blast_coords(self, x, y):
        blast_coords = [(x, y)]
        for i in range(1, 3):
            if self.new_game_state["field"][x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, 3):
            if self.new_game_state["field"][x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, 3):
            if self.new_game_state["field"][x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, 3):
            if self.new_game_state["field"][x, y - i] == -1:
                break
            blast_coords.append((x, y - i))
        return blast_coords
    
    def step_bombs(self, bombs):
        reward = 0
        for bomb in bombs:
            if bomb[1] == 0:
                blast_coords = self.get_blast_coords(*bomb[0])
                for coord in blast_coords:
                    if self.new_game_state["field"][coord] == 1:
                        self.new_game_state["field"][coord] = 0
                        reward += 0.1
                    if self.new_game_state["self"][3] == coord:
                        self.new_game_state["self"] = ("dead", self.new_game_state["self"][1], self.new_game_state["self"][2], self.new_game_state["self"][3])
                        reward -= 1
                    for other in self.new_game_state["others"]:
                        if other[3] == coord:
                            self.new_game_state["others"].remove(other)
                            reward += 1
                    self.new_game_state["explosion_map"][coord] = 2
            else:
                self.new_game_state["bombs"].append([bomb[0], bomb[1] - 1])
        return reward
    

    def evaluate_explosions(self):
        reward = 0
        for other in self.new_game_state["others"]:
            if self.new_game_state["explosion_map"][other[3]] > -1:
                self.new_game_state["others"].remove(other)
                reward += 1
        if self.new_game_state["explosion_map"][self.new_game_state["self"][3]] > -1:
            self.new_game_state["self"] = ("dead", self.new_game_state["self"][1], self.new_game_state["self"][2], self.new_game_state["self"][3])
            reward -= 1
        return reward
        

    def is_action_valid(self, game_state, action):
        self_position = game_state["self"][3]
        x, y = self_position
        if action == "UP":
            return game_state["field"][x, y - 1] == 0
        elif action == "DOWN":
            return game_state["field"][x, y + 1] == 0
        elif action == "LEFT":
            return game_state["field"][x - 1, y] == 0
        elif action == "RIGHT":
            return game_state["field"][x + 1, y] == 0
        elif action == "WAIT":
            return True
        elif action == "BOMB":
            return game_state["self"][2] == True
        else:
            raise ValueError(f"Invalid action: {action}")
