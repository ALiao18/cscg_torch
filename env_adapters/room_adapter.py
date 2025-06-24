# env_adapters/room_adapter.py
import numpy as np
import torch
from .base_adapter import CSCGEnvironmentAdapter

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

class RoomNPAdapter(CSCGEnvironmentAdapter):
    def __init__(self, room_array, no_up=[], no_down=[], no_left=[], no_right=[], start_pos=None, seed=42):
        super().__init__(seed=seed)
        self.room = room_array
        self.h, self.w = self.room.shape
        self.start_pos = start_pos
        self.no_up = set(no_up)
        self.no_down = set(no_down)
        self.no_left = set(no_left)
        self.no_right = set(no_right)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_map = ACTIONS
        self.n_actions = 4
        self.reset()

    def reset(self):
        if self.start_pos is None:
            free_positions = np.argwhere(self.room != -1)
            idx = self.rng.choice(len(free_positions))
            self.pos = tuple(free_positions[idx])
        else:
            self.pos = tuple(self.start_pos)
        return self.get_observation()

    def step(self, action):
        dr, dc = self.action_map[action]
        r, c = self.pos
        new_r, new_c = r + dr, c + dc

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
        # You can later switch this to a learned embedding
        up    = self.room[r - 1, c] != -1 if r > 0     else 0
        down  = self.room[r + 1, c] != -1 if r < self.h - 1 else 0
        left  = self.room[r, c - 1] != -1 if c > 0     else 0
        right = self.room[r, c + 1] != -1 if c < self.w - 1 else 0
        obs = (up << 3) + (down << 2) + (left << 1) + right
        return obs

class RoomTorchAdapter(CSCGEnvironmentAdapter):
    def __init__(self, room_tensor, no_up=[], no_down=[], no_left=[], no_right=[], start_pos=None, seed=42):
        super().__init__(seed=seed)
        # Ensure room tensor is on the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.room = room_tensor.to(self.device)
        self.h, self.w = self.room.shape
        self.start_pos = start_pos
        self.no_up = set(no_up)
        self.no_down = set(no_down)
        self.no_left = set(no_left)
        self.no_right = set(no_right)

        self.action_map = ACTIONS
        self.n_actions = 4
        self.reset()

    def reset(self):
        if self.start_pos is None:
            free_positions = (self.room != -1).nonzero(as_tuple=False)
            idx = self.rng.choice(len(free_positions))
            self.pos = tuple(free_positions[idx].tolist())
        else:
            self.pos = tuple(self.start_pos)
        return self.get_observation()

    def step(self, action):
        dr, dc = self.action_map[action]
        r, c = self.pos
        new_r, new_c = r + dr, c + dc

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
        # can later switch this to a learned embedding
        up    = self.room[r - 1, c] != -1 if r > 0     else 0
        down  = self.room[r + 1, c] != -1 if r < self.h - 1 else 0
        left  = self.room[r, c - 1] != -1 if c > 0     else 0
        right = self.room[r, c + 1] != -1 if c < self.w - 1 else 0
        obs = (up << 3) + (down << 2) + (left << 1) + right
        return obs
