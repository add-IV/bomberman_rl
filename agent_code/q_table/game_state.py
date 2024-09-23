from dataclasses import dataclass
import numpy as np
from enum import Enum


def state_to_index(game_state):
    features = state_to_features(game_state)
    return features.use_symmetry().index()


def state_to_features(game_state):
    game_state = GameState(**game_state)

    self_position = game_state.self[3]
    field = game_state.field
    coins = np.array(game_state.coins)
    crates = np.argwhere(field == 1)
    player_positions = np.array([player[3] for player in game_state.others])

    blocked = np.zeros(4, dtype=bool)

    if field[self_position[0], self_position[1] - 1] != 0:
        blocked[0] = True
    if field[self_position[0] + 1, self_position[1]] != 0:
        blocked[1] = True
    if field[self_position[0], self_position[1] + 1] != 0:
        blocked[2] = True
    if field[self_position[0] - 1, self_position[1]] != 0:
        blocked[3] = True

    game_state.explosion_map += bombs_explosion_map(game_state.bombs, game_state.field)

    danger = np.zeros(4, dtype=bool)
    if game_state.explosion_map[self_position[0], self_position[1] - 1] > 0:
        danger[0] = True
    if game_state.explosion_map[self_position[0] + 1, self_position[1]] > 0:
        danger[1] = True
    if game_state.explosion_map[self_position[0], self_position[1] + 1] > 0:
        danger[2] = True
    if game_state.explosion_map[self_position[0] - 1, self_position[1]] > 0:
        danger[3] = True

    current_danger = game_state.explosion_map[self_position[0], self_position[1]] > 0

    distance_nearest_coin = np.inf
    direction_nearest_coin = Direction.UP
    for coin in coins:
        distance = np.linalg.norm(self_position - coin)
        if distance < distance_nearest_coin:
            distance_nearest_coin = distance
            direction_nearest_coin = get_direction(self_position, coin)

    distance_nearest_crate = np.inf
    direction_nearest_crate = Direction.UP
    for crate in crates:
        distance = np.linalg.norm(self_position - crate)
        if distance < distance_nearest_crate:
            distance_nearest_crate = distance
            direction_nearest_crate = get_direction(self_position, crate)

    distance_nearest_player = np.inf
    direction_nearest_player = Direction.UP
    for player in player_positions:
        distance = np.linalg.norm(self_position - player)
        if distance < distance_nearest_player:
            distance_nearest_player = distance
            direction_nearest_player = get_direction(self_position, player)

    if distance_nearest_coin < 3:
        nearest_target_dir = direction_nearest_coin
        nearest_target_kind = TargetKind.COIN
    elif distance_nearest_crate < 3:
        nearest_target_dir = direction_nearest_crate
        nearest_target_kind = TargetKind.CRATE
    elif distance_nearest_player < 3:
        nearest_target_dir = direction_nearest_player
        nearest_target_kind = TargetKind.PLAYER
    else:
        nearest = np.argmin(
            [distance_nearest_coin, distance_nearest_crate, distance_nearest_player]
        )
        if nearest == 0:
            nearest_target_dir = direction_nearest_coin
            nearest_target_kind = TargetKind.COIN
        elif nearest == 1:
            nearest_target_dir = direction_nearest_crate
            nearest_target_kind = TargetKind.CRATE
        else:
            nearest_target_dir = direction_nearest_player
            nearest_target_kind = TargetKind.PLAYER

    if np.logical_or(blocked, danger).all() and current_danger:
        nearest_target_kind = TargetKind.SAFE_SPACE
        success_condition = standard_success_condition(game_state.explosion_map)
        abort_condition = standard_abort_condition(game_state.field)
        nearest_target_pos = bfs(self_position, abort_condition, success_condition)
        if nearest_target_pos:
            nearest_target_dir = get_direction(self_position, nearest_target_pos)
        else:
            nearest_target_dir = Direction.UP

    return FeatureState(
        blocked=blocked,
        danger=danger,
        current_danger=current_danger,
        nearest_target_dir=nearest_target_dir,
        nearest_target_kind=nearest_target_kind,
    )


def get_direction(start, end):
    # returns the longest direction from start to end
    x_diff = end[0] - start[0]
    y_diff = end[1] - start[1]
    if abs(x_diff) > abs(y_diff):
        if x_diff > 0:
            return Direction.RIGHT
        return Direction.LEFT
    if y_diff > 0:
        return Direction.DOWN
    return Direction.UP


def bfs(start, abort_condition, success_condition, get_path=False):
    """Breadth-first search with custom abort and success conditions"""
    queue = [start]
    visited = set()
    visited.add(start)
    parent = {}
    while queue:
        current = queue.pop(0)
        if success_condition(current):
            if get_path:
                path = []
                while current:
                    path.append(current)
                    current = parent.get(current)
                return path[::-1]
            return current
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            next_pos = tuple(np.add(current, translate_direction(direction)))
            if abort_condition(next_pos):
                continue
            if next_pos not in visited:
                visited.add(next_pos)
                parent[next_pos] = current
                queue.append(next_pos)
    return None


def standard_abort_condition(field):
    """basic abort condition: don't go into a wall"""
    return lambda next_pos: field[next_pos] == -1 or field[next_pos] == 1


def standard_success_condition(explosion_map):
    """basic success condition: next position is a coin"""
    return lambda next_pos: explosion_map[next_pos] == 0


def translate_direction(direction):
    """Translate a direction to a vector"""
    # TODO: make sure these directions are in sync with the ones in the environment
    if direction == "UP":
        return 0, -1
    if direction == "DOWN":
        return 0, 1
    if direction == "LEFT":
        return -1, 0
    if direction == "RIGHT":
        return 1, 0
    return 0, 0


def get_blast_coords(field, x, y) -> list[tuple[int, int]]:
    blast_coords = [(x, y)]
    for i in range(1, 3):
        if field[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, 3):
        if field[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, 3):
        if field[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, 3):
        if field[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))
    return blast_coords


def bombs_explosion_map(bombs, field) -> int:
    explosion_map = np.zeros_like(field)
    for bomb in bombs:
        blast_coords = get_blast_coords(field, *bomb[0])
        for coord in blast_coords:
            explosion_map[coord] = 2

    return explosion_map


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


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def next(self):
        return Direction((self.value + 1) % 4)


class TargetKind(Enum):
    COIN = 0
    CRATE = 1
    PLAYER = 2
    SAFE_SPACE = 3


@dataclass
class FeatureState:
    blocked: np.array
    danger: np.array
    current_danger: bool
    nearest_target_dir: Direction
    nearest_target_kind: TargetKind

    def rotate(self):
        return FeatureState(
            blocked=np.roll(self.blocked, 1),
            danger=np.roll(self.danger, 1),
            current_danger=self.current_danger,
            nearest_target_dir=self.nearest_target_dir.next(),
            nearest_target_kind=self.nearest_target_kind,
        )

    def flip(self):
        return FeatureState(
            blocked=np.roll(np.flip(self.blocked), 1),
            danger=np.roll(np.flip(self.danger), 1),
            current_danger=self.current_danger,
            nearest_target_dir=self.nearest_target_dir.next().next(),
            nearest_target_kind=self.nearest_target_kind,
        )

    def use_symmetry(self):
        index_blocked = np.where(self.blocked)[0]
        if len(index_blocked) == 0:
            return self
        if len(index_blocked) == 4:
            return self
        if len(index_blocked) == 1:
            if index_blocked[0] == 3:
                return self.rotate()
            if index_blocked[0] == 2:
                return self.rotate().rotate()
            if index_blocked[0] == 1:
                return self.rotate().rotate().rotate()
            if index_blocked[0] == 0:
                return self
        if len(index_blocked) == 3:
            if 0 not in index_blocked:
                return self.rotate().rotate().rotate()
            if 1 not in index_blocked:
                return self.rotate().rotate()
            if 2 not in index_blocked:
                return self.rotate()
            if 3 not in index_blocked:
                return self

        # len(index_blocked) == 2
        first_blocked = index_blocked[0]
        second_blocked = index_blocked[1]
        result = self
        # rotate to the first blocked
        if first_blocked == 2:
            result = result.rotate().rotate()
        if first_blocked == 1:
            result = result.rotate()
        if second_blocked == first_blocked + 1:
            return result
        return result.flip()

    @staticmethod
    def feature_size():
        return (2**4) * (2**4) * 2 * 4 * 4

    def index(self):
        index = 0
        for i, blocked in enumerate(self.blocked):
            index += blocked * (2**i)
        for i, danger in enumerate(self.danger):
            index += danger * (2 ** (i + 4))
        index += self.current_danger * (2**8)
        index += self.nearest_target_dir.value * (2**9)
        index += self.nearest_target_kind.value * (2**11)
        return index


if __name__ == "__main__":
    state = FeatureState(
        blocked=np.array([True, True, True, True]),
        danger=np.array([True, True, True, True]),
        current_danger=True,
        nearest_target_dir=Direction.LEFT,
        nearest_target_kind=TargetKind.SAFE_SPACE,
    )
    print(state.index())
    print(FeatureState.feature_size())
# end main
