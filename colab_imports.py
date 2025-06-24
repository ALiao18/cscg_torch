"""
Google Colab Import Helper

Simplified imports for Google Colab to avoid circular import issues.
"""

import torch
import numpy as np

def setup_colab_imports():
    """
    Setup imports for Google Colab environment.
    Returns a namespace with all necessary classes and functions.
    """
    # Import core model
    try:
        from cscg_torch.models.chmm_torch import CHMM_torch
        from cscg_torch.models import train_utils
    except ImportError:
        from models.chmm_torch import CHMM_torch
        from models import train_utils
    
    # Import base adapter
    try:
        from cscg_torch.env_adapters.base_adapter import CSCGEnvironmentAdapter
    except ImportError:
        from env_adapters.base_adapter import CSCGEnvironmentAdapter
    
    # Define RoomTorchAdapter directly to avoid import issues
    class RoomTorchAdapter(CSCGEnvironmentAdapter):
        """GPU-compatible room adapter for CHMM."""
        
        def __init__(self, room_tensor, no_up=[], no_down=[], no_left=[], no_right=[], start_pos=None, seed=42):
            super().__init__(seed=seed)
            # Parent class already sets self.device, so we can use it directly
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
    
    # Create namespace object
    class CSCGNamespace:
        def __init__(self):
            self.CHMM_torch = CHMM_torch
            self.RoomTorchAdapter = RoomTorchAdapter
            self.create_room_adapter = create_room_adapter
            self.get_room_n_clones = get_room_n_clones
            self.train_utils = train_utils
            
            # Add all training functions
            self.validate_seq = train_utils.validate_seq
            self.forward = train_utils.forward
            self.backward = train_utils.backward
            self.forward_mp = train_utils.forward_mp
            self.backtrace = train_utils.backtrace
    
    return CSCGNamespace()

# Auto-setup for convenience
cscg = setup_colab_imports()

# Make everything available at module level
CHMM_torch = cscg.CHMM_torch
RoomTorchAdapter = cscg.RoomTorchAdapter
create_room_adapter = cscg.create_room_adapter
get_room_n_clones = cscg.get_room_n_clones