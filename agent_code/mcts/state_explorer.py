import numpy as np
import copy
from dataclasses import dataclass


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]

@dataclass
class GameState:
    self: tuple[str, int, bool, tuple[int, int]]
    others: list[tuple[str, int, bool, tuple[int, int]]]
    coins: list[tuple[int, int]]
    bombs: list[tuple[tuple[int, int], int]]
    field: np.ndarray
    step: int
    explosion_map: np.ndarray

    round: int
    user_input: str
    bomb_cd: int = 0

class StateExplorer:
    def __init__(self):
        self.new_game_state = None
        self.old_game_state = None

    def is_terminal(self, game_state: GameState) -> bool:
        # either the player is dead or the game is over
        # game is over if no coins, crates, or players are left
        if game_state.self[0] == "dead":
            return True
        others_dead = game_state.others == []
        coins = game_state.coins != []
        crates = 1 in game_state.field
        if others_dead and not coins and not crates:
            return True
        return False
    
    def is_dead(self, game_state: GameState) -> bool:
        return game_state.self[0] == "dead"

    def get_valid_actions(self, game_state: GameState) -> list[str]:
        return [
            action for action in ACTIONS if self.is_action_valid(game_state, action)
        ]

    def get_score(self, game_state: GameState) -> int:
        return game_state.self[1]

    def step(self, game_state: GameState, action) -> GameState:
        if not self.is_action_valid(game_state, action):
            raise ValueError(f"Invalid action: {action}")
        
        self.new_game_state = copy.deepcopy(game_state)
        self.old_game_state = game_state

        self.cull_far_others()

        reward = 0

        self.step_player(action)
        reward += self.disincentivize_waiting(action)
        self.step_others()
        reward += self.step_coins()
        reward += self.step_bombs()
        reward += self.evaluate_explosions()
        self.step_explosion_map()

        return self.new_game_state, reward
    
    def no_close_bombs(self) -> bool:
        all_far = True
        for bomb in self.new_game_state.bombs:
            if abs(bomb[0][0] - self.new_game_state.self[3][0]) < 4 and abs(bomb[0][1] - self.new_game_state.self[3][1]) < 4:
                all_far = False
        return all_far
    
    def disincentivize_waiting(self, action) -> int:
        if action == "WAIT" and self.no_close_bombs():
            return -0.1
        return 0
        
    def cull_far_others(self) -> None:
        close_others = []
        for other in self.new_game_state.others:
            if abs(other[3][0] - self.new_game_state.self[3][0]) < 4 and abs(other[3][1] - self.new_game_state.self[3][1]) < 4:
                close_others.append(other)
        self.new_game_state.others = close_others
    
    def step_player(self, action) -> None:
        self_position = self.old_game_state.self[3]
        can_bomb = self.old_game_state.self[2]
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
                self.new_game_state.bombs.append([self_position, 5])
                can_bomb = False
                self.new_game_state.bomb_cd = 4
        else:
            raise ValueError(f"Invalid action: {action}")
        self.new_game_state.bomb_cd = max(0, self.old_game_state.bomb_cd - 1)
        if self.new_game_state.bomb_cd == 0:
            can_bomb = True
        self.new_game_state.self = (
            self.old_game_state.self[0],
            self.old_game_state.self[1],
            can_bomb,
            self_position,
        )
        
    def step_others(self) -> None:
        pass

    def step_coins(self) -> int:
        reward = 0
        coins = self.new_game_state.coins
        self.new_game_state.coins = []
        for coin in coins:
            collected = False
            if coin == self.new_game_state.self[3]:
                reward += 0.2
                collected = True
            for other in self.new_game_state.others:
                if coin == other[3]:
                    collected = True
            if not collected:
                self.new_game_state.coins.append(coin)
        return reward
    
    def step_explosion_map(self) -> None:
        self.new_game_state.explosion_map = np.clip(self.new_game_state.explosion_map - 1, 0, 2)

    def get_blast_coords(self, x, y) -> list[tuple[int, int]]:
        blast_coords = [(x, y)]
        for i in range(1, 3):
            if self.new_game_state.field[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, 3):
            if self.new_game_state.field[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, 3):
            if self.new_game_state.field[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, 3):
            if self.new_game_state.field[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))
        return blast_coords
    
    def step_bombs(self) -> int:
        reward = 0
        bombs = self.new_game_state.bombs
        self.new_game_state.bombs = []
        for bomb in bombs:
            if bomb[1] == 0:
                blast_coords = self.get_blast_coords(*bomb[0])
                for coord in blast_coords:
                    if self.new_game_state.field[coord] == 1:
                        self.new_game_state.field[coord] = 0
                        reward += 0.1
                    self.new_game_state.explosion_map[coord] = 2
            else:
                self.new_game_state.bombs.append([bomb[0], bomb[1] - 1])
        return reward
    

    def evaluate_explosions(self) -> int:
        reward = 0
        for other in self.new_game_state.others:
            if self.new_game_state.explosion_map[other[3]] > 0:
                self.new_game_state.others.remove(other)
                reward += 5
        if self.new_game_state.explosion_map[self.new_game_state.self[3]] > 0:
            self.new_game_state.self = ("dead", self.new_game_state.self[1], self.new_game_state.self[2], self.new_game_state.self[3])
            reward -= 5
        return reward
        

    def is_action_valid(self, game_state, action) -> bool:
        self_position = game_state.self[3]
        x, y = self_position
        if action == "UP":
            return game_state.field[x, y - 1] == 0
        elif action == "DOWN":
            return game_state.field[x, y + 1] == 0
        elif action == "LEFT":
            return game_state.field[x - 1, y] == 0
        elif action == "RIGHT":
            return game_state.field[x + 1, y] == 0
        elif action == "WAIT":
            return True
        elif action == "BOMB":
            return game_state.self[2] is True
        else:
            raise ValueError(f"Invalid action: {action}")


if __name__ == "__main__":
    state1 = GameState(
        self=("alive", 1, True, (1, 1)),
        others=[("alive", 1, True, (2, 1))],
        coins=[(2, 2)],
        bombs=[],
        field=np.zeros((5, 5)),
        step=0,
        explosion_map=np.zeros((5, 5)),
        round=0,
        user_input="",
    )
    state2 = copy.deepcopy(state1)
    state3 = copy.copy(state1)

    state1.field[2, 2] = 1

    print(state2.field)
    print(state3.field)
    # needs to use deepcopy

# end main