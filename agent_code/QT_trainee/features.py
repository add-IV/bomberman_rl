from .constants import ACTIONS
import numpy as np

class BombermanFeatureExtractor:
    def __init__(self, game_state):
        self.game = game_state

    def num_features(self):
        return 5  # or the actual number of features you want to extract

    def num_actions(self):
        return len(ACTIONS)

    def nearest_bomb_timer(self, game_state, player_pos):
        """
        Get the countdown timer of the nearest bomb.
        """
        bombs = game_state['bombs']
        if not bombs:
            return 0  # No bomb nearby
        distances = [(abs(player_pos[0] - bx) + abs(player_pos[1] - by), timer) for (bx, by), timer in bombs]
        nearest_bomb = min(distances, key=lambda x: x[0])
        return nearest_bomb[1]  # Return the timer of the nearest bomb

    def is_in_bomb_danger(self, game_state, player_pos):
        """
        Check if the agent is in the danger zone of any active bomb.
        """
        bombs = game_state['bombs']
        for (bx, by), timer in bombs:
            if abs(player_pos[0] - bx) + abs(player_pos[1] - by) <= 4:  # Assuming bomb affects a 4-square radius
                return 1  # Danger present
        return 0  # No danger

    def encode_state(self, game_state):
        """
        Encode the current game state into features.
        """
        player_pos = game_state['self'][3]  # Extract player position (x, y)
        bomb_timer = self.nearest_bomb_timer(game_state, player_pos)
        is_escape_phase = self.is_in_bomb_danger(game_state, player_pos)

        # Blocked squares for directions (UP, RIGHT, DOWN, LEFT)
        blocked_squares = [
            self.is_blocked(game_state, player_pos, 'UP'),
            self.is_blocked(game_state, player_pos, 'RIGHT'),
            self.is_blocked(game_state, player_pos, 'DOWN'),
            self.is_blocked(game_state, player_pos, 'LEFT')
        ]

        # Explosion danger for directions (UP, RIGHT, DOWN, LEFT)
        explosion_danger = [
            self.is_explosion_near(game_state, player_pos, 'UP'),
            self.is_explosion_near(game_state, player_pos, 'RIGHT'),
            self.is_explosion_near(game_state, player_pos, 'DOWN'),
            self.is_explosion_near(game_state, player_pos, 'LEFT')
        ]

        # Direction to nearest coin, crate, and safe square
        nearest_coin_direction = self.direction_to_nearest(player_pos, game_state['coins'])
        nearest_crate_direction = self.direction_to_nearest(player_pos, game_state['field'], target_type='crate')
        nearest_safe_square_direction = self.direction_to_nearest_safe_square(player_pos, game_state)

        # Combine features into a tuple representing the state
        state = (
            tuple(blocked_squares),
            tuple(explosion_danger),
            nearest_coin_direction,
            nearest_crate_direction,
            nearest_safe_square_direction,
            bomb_timer,
            is_escape_phase
        )

        return state

    def is_explosion_near(self, game_state, player_pos, direction):
        """
        Check if there's an imminent explosion near the player in the given direction.
        Returns 1 if there's an explosion danger, 0 otherwise.
        """
        x, y = player_pos
        explosion_map = game_state['explosion_map']  # Assuming explosion_map represents imminent explosions

        # Determine the position to check based on the direction
        if direction == 'UP':
            new_pos = (x, y - 1)
        elif direction == 'RIGHT':
            new_pos = (x + 1, y)
        elif direction == 'DOWN':
            new_pos = (x, y + 1)
        elif direction == 'LEFT':
            new_pos = (x - 1, y)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Check if the new position is within bounds
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= explosion_map.shape[0] or new_pos[1] >= \
                explosion_map.shape[1]:
            return 0  # No explosion danger if out of bounds

        # Check if there's an explosion in the new position
        if explosion_map[new_pos[0], new_pos[1]] > 0:  # Assuming non-zero values indicate an explosion
            return 1  # Explosion danger

        return 0  # No explosion danger

    def direction_to_nearest(self, player_pos, targets, target_type=None):
        """
        Calculate the direction to the nearest target (coin, crate, safe square).
        Return an encoded direction:
        - 0: No target nearby
        - 1: Top
        - 2: Right
        - 3: Bottom
        - 4: Left
        """
        # If targets is a NumPy array, use .size to check if it's empty
        if isinstance(targets, np.ndarray):
            if targets.size == 0:
                return 0  # No targets
        else:  # Otherwise, if it's a list or other iterable
            if len(targets) == 0:
                return 0  # No targets

        # Calculate Manhattan distances to all targets
        distances = [(abs(player_pos[0] - t[0]) + abs(player_pos[1] - t[1]), t) for t in targets]
        nearest_target = min(distances, key=lambda x: x[0])[1]  # Find the nearest target

        # Determine direction
        if nearest_target[1] < player_pos[1]:
            return 1  # Top
        elif nearest_target[0] > player_pos[0]:
            return 2  # Right
        elif nearest_target[1] > player_pos[1]:
            return 3  # Bottom
        elif nearest_target[0] < player_pos[0]:
            return 4  # Left

        return 0  # No valid target found

    def is_blocked(self, game_state, player_pos, direction):
        """
        Check if the movement in the given direction is blocked by an obstacle (e.g., wall, crate).
        Returns 1 if blocked, 0 otherwise.
        """
        x, y = player_pos
        field = game_state['field']  # Assuming the field represents obstacles (walls, crates)

        # Determine the new position based on the direction
        if direction == 'UP':
            new_pos = (x, y - 1)
        elif direction == 'RIGHT':
            new_pos = (x + 1, y)
        elif direction == 'DOWN':
            new_pos = (x, y + 1)
        elif direction == 'LEFT':
            new_pos = (x - 1, y)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Check if new position is within bounds
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= field.shape[0] or new_pos[1] >= field.shape[1]:
            return 1  # Blocked if out of bounds

        # Check if the new position is blocked by a wall or crate
        if field[new_pos[0], new_pos[1]] != 0:  # Assuming 0 means free space
            return 1  # Blocked by wall or crate

        # You can add further checks here for bombs or other obstacles
        return 0  # Not blocked

    def direction_to_nearest_safe_square(self, player_pos, game_state):
        """
        Calculate the direction to the nearest safe square.
        Return an encoded direction:
        - 0: No safe square nearby
        - 1: Top
        - 2: Right
        - 3: Bottom
        - 4: Left
        """
        field = game_state['field']
        bombs = game_state['bombs']

        # Get all the possible positions on the field
        possible_positions = [(x, y) for x in range(len(field)) for y in range(len(field[0]))]

        # Filter out unsafe squares
        unsafe_positions = set()

        for bomb in bombs:
            bx, by = bomb[0]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = bx + dx, by + dy
                    if 0 <= nx < len(field) and 0 <= ny < len(field[0]):
                        unsafe_positions.add((nx, ny))

        safe_positions = [pos for pos in possible_positions if
                          pos not in unsafe_positions and field[pos[0]][pos[1]] == 0]

        if not safe_positions:
            return 0  # No safe positions

        # Calculate Manhattan distances to all safe positions
        distances = [(abs(player_pos[0] - t[0]) + abs(player_pos[1] - t[1]), t) for t in safe_positions]
        nearest_safe = min(distances, key=lambda x: x[0])[1]  # Find the nearest safe square

        # Determine direction
        if nearest_safe[1] < player_pos[1]:
            return 1  # Top
        elif nearest_safe[0] > player_pos[0]:
            return 2  # Right
        elif nearest_safe[1] > player_pos[1]:
            return 3  # Bottom
        elif nearest_safe[0] < player_pos[0]:
            return 4  # Left

        return 0  # No valid target found

