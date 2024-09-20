import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nine_by_nine_to_six_by_six(board: np.ndarray) -> np.ndarray:
    """
    shape:
    000010000
    000111000
    000111000
    011111110
    111101111
    011111110
    000111000
    000111000
    000010000
    indexes:
    0,1,2,3,3,4,
    1,1,2,3,4,5,
    2,3,3,4,4,5,
    3,4,4,5,5,6,
    3,4,5,6,7,7,
    4,5,5,6,7,8,
    """
    indexes_x = np.array(
        [
            [0, 1, 2, 3, 3, 4],
            [1, 1, 2, 3, 4, 5],
            [2, 3, 3, 4, 4, 5],
            [3, 4, 4, 5, 5, 6],
            [3, 4, 5, 6, 7, 7],
            [4, 5, 5, 6, 7, 8],
        ]
    ).flatten()
    indexes_y = np.array(
        [
            [4, 5, 5, 6, 7, 8],
            [3, 4, 4, 5, 7, 7],
            [3, 3, 4, 5, 6, 6],
            [2, 2, 3, 4, 5, 5],
            [1, 1, 3, 4, 4, 5],
            [0, 1, 2, 3, 3, 4],
        ]
    ).flatten()
    return board[indexes_x, indexes_y].reshape(6, 6)


def state_from_game_state(game_state):
    # get the relevant information from the game state
    board = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    self = game_state["self"]
    self_bomb_available = self[2]
    self_position = self[3]
    others = game_state["others"]

    # initialize 9x9 state arrays
    view_range = 4
    nine_by_nine_state = np.zeros((7, 9, 9))
    padded_board = np.pad(board, view_range, constant_values=-1)
    padded_explosion_map = np.pad(explosion_map, view_range, constant_values=0)

    x, y = self_position

    # split crates and walls
    nine_by_nine_state[0] = (
        padded_board[x : x + 2 * view_range + 1, y : y + 2 * view_range + 1] == -1
    )
    nine_by_nine_state[1] = (
        padded_board[x : x + 2 * view_range + 1, y : y + 2 * view_range + 1] == 1
    )

    # coins
    for coin in coins:
        coin_x, coin_y = coin
        if (
            x - view_range <= coin_x <= x + view_range
            and y - view_range <= coin_y <= y + view_range
        ):
            nine_by_nine_state[2, coin_x - x + view_range, coin_y - y + view_range] = 1

    # bombs
    for bomb in bombs:
        bomb_x, bomb_y = bomb[0]
        bomb_timer = bomb[1]
        if (
            x - view_range <= bomb_x <= x + view_range
            and y - view_range <= bomb_y <= y + view_range
        ):
            nine_by_nine_state[3, bomb_x - x + view_range, bomb_y - y + view_range] = (
                bomb_timer / 4
            )  # TODO: check if this is correct

    # explosions
    nine_by_nine_state[4] = (
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
            nine_by_nine_state[
                5, player_x - x + view_range, player_y - y + view_range
            ] = (2 if bomb_available else 1)

    # self
    nine_by_nine_state[6, :, :] = self_bomb_available

    six_by_six_state = np.zeros((7, 6, 6))

    for i in range(7):
        six_by_six_state[i] = nine_by_nine_to_six_by_six(nine_by_nine_state[i])

    return torch.tensor(six_by_six_state, dtype=torch.float32).unsqueeze(0).to(device)


if __name__ == "__main__":
    state = state_from_game_state(
        {
            "round": 1,
            "step": 1,
            "field": np.zeros((11, 11)),
            "coins": [(1, 1), (2, 2), (3, 3)],
            "bombs": [((1, 1), 3), ((2, 2), 2), ((3, 3), 1)],
            "explosion_map": np.zeros((11, 11)),
            "self": (0, 0, 1, (4, 4)),
            "others": [(0, 0, 1, (5, 5)), (0, 0, 1, (6, 6)), (0, 0, 1, (7, 7))],
        }
    )
    print(state)
    print(state.shape)
# end main
