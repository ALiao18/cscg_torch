import numpy as np
import torch

class CSCGEnvironmentAdapter:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def is_terminal(self):
        return False

    def generate_sequence(self, length):
        x_seq, a_seq = [], []
        self.reset()
        for _ in range(length):
            obs = self.get_observation()
            action = self.rng.choice(self.n_actions)
            new_obs, valid = self.step(action)
            if valid:
                x_seq.append(obs)
                a_seq.append(action)
        return np.array(x_seq), np.array(a_seq)