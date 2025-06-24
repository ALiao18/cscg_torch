import numpy as np
import torch

class CSCGEnvironmentAdapter:
    def __init__(self, seed=42):
        # Strict type assertions for initialization
        assert isinstance(seed, int), f"seed must be int, got {type(seed)}"
        assert seed >= 0, f"seed must be non-negative, got {seed}"
        
        self.rng = np.random.RandomState(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = None  # Should be set by subclasses
        
        # Post-initialization assertions
        assert isinstance(self.rng, np.random.RandomState), f"rng must be RandomState, got {type(self.rng)}"
        assert isinstance(self.device, torch.device), f"device must be torch.device, got {type(self.device)}"

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def is_terminal(self):
        return False

    def generate_sequence(self, length):
        # Strict input validation
        assert isinstance(length, int), f"length must be int, got {type(length)}"
        assert length > 0, f"length must be positive, got {length}"
        assert self.n_actions is not None, "n_actions must be set by subclass"
        assert isinstance(self.n_actions, int), f"n_actions must be int, got {type(self.n_actions)}"
        assert self.n_actions > 0, f"n_actions must be positive, got {self.n_actions}"
        
        x_seq, a_seq = [], []
        self.reset()
        
        for i in range(length):
            obs = self.get_observation()
            
            # Strict type checking for observation
            assert obs is not None, f"get_observation() returned None at step {i}"
            assert isinstance(obs, (int, np.integer)), f"obs must be int, got {type(obs)} at step {i}"
            
            action = self.rng.choice(self.n_actions)
            
            # Strict type checking for action
            assert isinstance(action, (int, np.integer)), f"action must be int, got {type(action)} at step {i}"
            assert 0 <= action < self.n_actions, f"action {action} out of range [0, {self.n_actions}) at step {i}"
            
            new_obs, valid = self.step(action)
            
            # Strict type checking for step results
            assert isinstance(valid, bool), f"step() must return bool valid, got {type(valid)} at step {i}"
            
            if valid:
                # Ensure types are safe for numpy array conversion
                x_seq.append(int(obs))
                a_seq.append(int(action))
        
        # Final output validation
        result_x = np.array(x_seq, dtype=np.int64)
        result_a = np.array(a_seq, dtype=np.int64)
        
        assert len(result_x) == len(result_a), f"Sequence lengths mismatch: x={len(result_x)}, a={len(result_a)}"
        assert len(result_x) <= length, f"Generated sequence too long: {len(result_x)} > {length}"
        
        return result_x, result_a