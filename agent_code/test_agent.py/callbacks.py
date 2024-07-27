import numpy as np


def setup(self):
    self.i = 5


def act(self, game_state: dict):
    self.i += 1
    self.logger.info('Pick action at random')
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
