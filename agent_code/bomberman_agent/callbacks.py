import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def setup(self):
    """Setup code, called once at the start of the game"""
    self.i = 5


def standard_abort_condition(field):
    """basic abort condition: don't go into a wall"""
    return lambda next_pos: field[next_pos] == -1 or field[next_pos] == 1


def standard_success_condition(coins):
    """basic success condition: next position is a coin"""
    return lambda next_pos: next_pos in coins


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

def translate_to_direction(vector):
    """Translate a vector to a direction"""
    if vector[0] == 0 and vector[1] == -1:
        return "UP"
    if vector[0] == 0 and vector[1] == 1:
        return "DOWN"
    if vector[0] == -1 and vector[1] == 0:
        return "LEFT"
    if vector[0] == 1 and vector[1] == 0:
        return "RIGHT"
    return "WAIT"


def act(self, game_state: dict):
    """Act based on game state, once per game step"""
    # currently uses bfs to find the nearest coin and moves towards it
    json_game_state = json.dumps(game_state, cls=NumpyEncoder)
    self.logger.debug(f"Game state: {json_game_state}")
    self.i += 1
    # find nearest coin
    coins: list[tuple[int, int]] = game_state["coins"]
    field = game_state["field"]
    agent: tuple[str, int, bool, tuple[int, int]] = game_state["self"]
    agent_location = agent[3]
    abort_condition = standard_abort_condition(field)
    success_condition = standard_success_condition(coins)
    path_to_coin = bfs(
        agent_location, abort_condition, success_condition, get_path=True
    )
    self.logger.debug(f"Coin amount: {len(coins)}")
    self.logger.debug(f"location: {agent_location}")
    self.logger.debug(f"Path to nearest coin: {path_to_coin}")
    self.logger.debug(f"Nearest coin: {path_to_coin[-1]}")
    if len(path_to_coin) < 2:
        self.logger.info("No path found, waiting")
        return "WAIT"
    next_step = path_to_coin[1]
    self.logger.debug(f"Next step: {next_step}")
    direction = np.subtract(next_step, agent_location)
    self.logger.debug(f"Direction: {direction}")
    action = translate_to_direction(direction)
    self.logger.info(f"Action: {action}")
    return action