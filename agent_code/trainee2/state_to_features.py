import numpy as np
def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector.
    """
    if game_state is None:
        return np.zeros(10)  # Example feature vector

    player_x, player_y = game_state['self'][3]
    coins = np.array(game_state['coins'])
    if len(coins) > 0:
        distances = np.linalg.norm(coins - np.array([player_x, player_y]), axis=1)
        nearest_coin_distance = np.min(distances)
    else:
        nearest_coin_distance = -1

    features = np.array([player_x, player_y, nearest_coin_distance])
    return features
