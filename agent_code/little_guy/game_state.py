import numpy as np
import torch

from agent_code.little_guy.state_explorer import GameState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seven_by_seven_to_four_by_four(board: np.ndarray) -> np.ndarray:
    """
    shape:
    00100
    01110
    11011
    01110
    00100
    """
    indexes_x = np.array(
        [
            [1, 0, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 4, 3],
        ]
    ).flatten()
    indexes_y = np.array(
        [
            [1, 2, 2, 3],
            [0, 1, 3, 4],
            [1, 2, 2, 3],
        ]
    ).flatten()
    return board[indexes_x, indexes_y].reshape(3, 4)


def state_from_game_state(game_state):
    # get the relevant information from the game state
    board = game_state.field
    coins = game_state.coins
    bombs = game_state.bombs
    explosion_map = game_state.explosion_map
    self = game_state.self
    self_bomb_available = self[2]
    self_position = self[3]
    others = game_state.others

    # initialize 5x5 state arrays
    view_range = 2
    seven_by_seven_state = np.zeros((7, 5, 5))
    padded_board = np.pad(board, view_range, constant_values=-1)
    padded_explosion_map = np.pad(explosion_map, view_range, constant_values=0)

    x, y = self_position

    # split crates and walls
    seven_by_seven_state[0] = (
        padded_board[x : x + 2 * view_range + 1, y : y + 2 * view_range + 1] == -1
    )
    seven_by_seven_state[1] = (
        padded_board[x : x + 2 * view_range + 1, y : y + 2 * view_range + 1] == 1
    )

    # coins
    for coin in coins:
        coin_x, coin_y = coin
        if (
            x - view_range <= coin_x <= x + view_range
            and y - view_range <= coin_y <= y + view_range
        ):
            seven_by_seven_state[
                2, coin_x - x + view_range, coin_y - y + view_range
            ] = 1

    # bombs
    for bomb in bombs:
        bomb_x, bomb_y = bomb[0]
        bomb_timer = bomb[1]
        if (
            x - view_range <= bomb_x <= x + view_range
            and y - view_range <= bomb_y <= y + view_range
        ):
            seven_by_seven_state[
                3, bomb_x - x + view_range, bomb_y - y + view_range
            ] = (
                bomb_timer / 4
            )  # TODO: check if this is correct

    # explosions
    seven_by_seven_state[4] = (
        padded_explosion_map[x : x + 2 * view_range + 1, y : y + 2 * view_range + 1] / 2
    )  # TODO: check if this is correct

    # players
    for player in others:
        player_x, player_y = player[3]
        bomb_available = player[2]
        if (
            x - view_range <= player_x <= x + view_range
            and y - view_range <= player_y <= y + view_range
        ):
            seven_by_seven_state[
                5, player_x - x + view_range, player_y - y + view_range
            ] = (2 if bomb_available else 1)

    # self
    seven_by_seven_state[6, :, :] = self_bomb_available

    four_by_four_state = np.zeros((7, 3, 4))

    for i in range(7):
        four_by_four_state[i] = seven_by_seven_to_four_by_four(seven_by_seven_state[i])

    return torch.tensor(four_by_four_state, dtype=torch.float32).unsqueeze(0).to(device)


if __name__ == "__main__":
    game_state = {
        "round": 1,
        "step": 1,
        "field": np.zeros((11, 11)),
        "coins": [(1, 1), (2, 2), (3, 3)],
        "bombs": [((1, 1), 3), ((2, 2), 2), ((3, 3), 1)],
        "explosion_map": np.zeros((11, 11)),
        "self": (0, 0, 1, (4, 4)),
        "others": [(0, 0, 1, (5, 5)), (0, 0, 1, (6, 6)), (0, 0, 1, (7, 7))],
        "user_input": "UP",
    }
    game_state = GameState(**game_state)
    state = state_from_game_state(game_state)
    print(state)
    print(state.shape)
# end main
