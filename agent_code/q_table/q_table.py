import numpy as np


def save_q_table(self):
    """
    Save the Q-table to a file.
    """
    np.save("my-saved-q-table.npy", self.q_table)


def load_q_table():
    """
    Load the Q-table from a file.
    """
    return np.load("my-saved-q-table.npy")
