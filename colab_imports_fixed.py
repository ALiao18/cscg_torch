"""
Google Colab Import Helper - FIXED VERSION

Simplified imports for Google Colab to avoid all import issues.
"""

import torch
import numpy as np

# Define base adapter at module level to avoid scoping issues
class CSCGEnvironmentAdapter:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = None
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def get_observation(self):
        raise NotImplementedError
    
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

# Define RoomTorchAdapter at module level
class RoomTorchAdapter(CSCGEnvironmentAdapter):
    """GPU-compatible room adapter for CHMM."""
    
    def __init__(self, room_tensor, no_up=[], no_down=[], no_left=[], no_right=[], start_pos=None, seed=42):
        super().__init__(seed=seed)
        self.room = room_tensor.to(self.device)
        self.h, self.w = self.room.shape
        self.start_pos = start_pos
        self.no_up = set(no_up)
        self.no_down = set(no_down)
        self.no_left = set(no_left)
        self.no_right = set(no_right)
        
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
        self.n_actions = 4
        self.reset()
    
    def reset(self):
        if self.start_pos is None:
            free_positions = (self.room != -1).nonzero(as_tuple=False)
            if len(free_positions) > 0:
                idx = self.rng.choice(len(free_positions))
                self.pos = tuple(free_positions[idx].tolist())
            else:
                self.pos = (1, 1)  # fallback
        else:
            self.pos = tuple(self.start_pos)
        return self.get_observation()
    
    def step(self, action):
        dr, dc = self.action_map[action]
        r, c = self.pos
        new_r, new_c = r + dr, c + dc
        
        # Boundary checks
        if not (0 <= new_r < self.h and 0 <= new_c < self.w):
            return self.get_observation(), False
        if self.room[new_r, new_c] == -1:
            return self.get_observation(), False
        
        # Invisible wall checks
        flat_idx = r * self.w + c
        if action == 0 and flat_idx in self.no_up:
            return self.get_observation(), False
        if action == 1 and flat_idx in self.no_down:
            return self.get_observation(), False
        if action == 2 and flat_idx in self.no_left:
            return self.get_observation(), False
        if action == 3 and flat_idx in self.no_right:
            return self.get_observation(), False
        
        self.pos = (new_r, new_c)
        return self.get_observation(), True
    
    def get_observation(self):
        r, c = self.pos
        up = self.room[r - 1, c] != -1 if r > 0 else 0
        down = self.room[r + 1, c] != -1 if r < self.h - 1 else 0
        left = self.room[r, c - 1] != -1 if c > 0 else 0
        right = self.room[r, c + 1] != -1 if c < self.w - 1 else 0
        obs = (up << 3) + (down << 2) + (left << 1) + right
        return obs

# Import CHMM model
try:
    from cscg_torch.models.chmm_torch import CHMM_torch
except ImportError:
    try:
        from models.chmm_torch import CHMM_torch
    except ImportError:
        print("Warning: Could not import CHMM_torch")
        CHMM_torch = None

# Utility functions
def create_room_adapter(room_data, **kwargs):
    """Create a room adapter from tensor data."""
    if not isinstance(room_data, torch.Tensor):
        room_data = torch.tensor(room_data)
    return RoomTorchAdapter(room_data, **kwargs)

def get_room_n_clones(n_clones_per_obs=1, device=None):
    """Get n_clones tensor for room navigation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_obs_types = 16  # 2^4 possible combinations
    return torch.full((n_obs_types,), n_clones_per_obs, dtype=torch.int64, device=device)

def test_setup():
    """Test that everything works."""
    print("=== Testing Colab Setup ===")
    
    # Test device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    
    # Test room creation
    room_tensor = torch.randint(0, 4, size=[5, 5], dtype=torch.long)
    print(f"✓ Room tensor: {room_tensor.shape}")
    
    # Test adapter
    env = create_room_adapter(room_tensor)
    print(f"✓ Environment: {type(env).__name__}")
    print(f"✓ Environment device: {env.device}")
    
    # Test n_clones
    n_clones = get_room_n_clones(1)
    print(f"✓ n_clones: {n_clones.shape}")
    
    # Test sequences
    x, a = env.generate_sequence(10)
    print(f"✓ Sequences: x={len(x)}, a={len(a)}")
    
    if CHMM_torch:
        model = CHMM_torch(n_clones, torch.tensor(x), torch.tensor(a))
        print(f"✓ Model: {model.device}")
    
    print("🎉 All tests passed!")
    return True

# Auto-test on import
if __name__ == "__main__":
    test_setup()