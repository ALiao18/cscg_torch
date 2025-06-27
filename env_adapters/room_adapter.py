# env_adapters/room_adapter.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from .base_adapter import CSCGEnvironmentAdapter
import os

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

class RoomNPAdapter(CSCGEnvironmentAdapter):
    def __init__(self, room_array, no_up=[], no_down=[], no_left=[], no_right=[], start_pos=None, seed=42):
        # Strict input validation
        assert isinstance(room_array, np.ndarray), f"room_array must be numpy array, got {type(room_array)}"
        assert room_array.ndim == 2, f"room_array must be 2D, got {room_array.ndim}D"
        assert room_array.dtype in [np.int32, np.int64], f"room_array must be integer type, got {room_array.dtype}"
        assert room_array.size > 0, "room_array cannot be empty"
        
        super().__init__(seed=seed)
        
        self.room = room_array
        self.h, self.w = int(self.room.shape[0]), int(self.room.shape[1])
        
        # Validate dimensions
        assert self.h > 0 and self.w > 0, f"Invalid room dimensions: {self.h}x{self.w}"
        
        # Validate start position
        if start_pos is not None:
            assert isinstance(start_pos, (tuple, list)), f"start_pos must be tuple/list, got {type(start_pos)}"
            assert len(start_pos) == 2, f"start_pos must have 2 elements, got {len(start_pos)}"
            r, c = start_pos
            assert isinstance(r, int) and isinstance(c, int), f"start_pos elements must be int, got {type(r)}, {type(c)}"
            assert 0 <= r < self.h and 0 <= c < self.w, f"start_pos {start_pos} out of bounds for {self.h}x{self.w} room"
        
        self.start_pos = start_pos
        
        # Validate wall lists
        for wall_list, name in [(no_up, "no_up"), (no_down, "no_down"), (no_left, "no_left"), (no_right, "no_right")]:
            assert isinstance(wall_list, (list, tuple, set)), f"{name} must be list/tuple/set, got {type(wall_list)}"
            for idx in wall_list:
                assert isinstance(idx, int), f"{name} elements must be int, got {type(idx)}"
                assert 0 <= idx < self.h * self.w, f"{name} index {idx} out of bounds for {self.h}x{self.w} room"
        
        self.no_up = set(no_up)
        self.no_down = set(no_down)
        self.no_left = set(no_left)
        self.no_right = set(no_right)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_map = ACTIONS
        self.n_actions = int(4)
        
        # Post-initialization validation
        assert isinstance(self.n_actions, int), f"n_actions must be int, got {type(self.n_actions)}"
        assert self.n_actions == len(ACTIONS), f"n_actions mismatch: {self.n_actions} != {len(ACTIONS)}"
        
        self.reset()

    def reset(self):
        # Strict validation for reset operation
        assert hasattr(self, 'room'), "room must be initialized"
        assert hasattr(self, 'rng'), "rng must be initialized"
        assert isinstance(self.room, np.ndarray), f"room must be numpy array, got {type(self.room)}"
        
        if self.start_pos is None:
            free_positions = np.argwhere(self.room != -1)
            assert len(free_positions) > 0, "No free positions found in room"
            assert isinstance(free_positions, np.ndarray), f"free_positions must be numpy array, got {type(free_positions)}"
            
            idx = self.rng.choice(len(free_positions))
            assert isinstance(idx, (int, np.integer)), f"idx must be int, got {type(idx)}"
            assert 0 <= idx < len(free_positions), f"idx {idx} out of range [0, {len(free_positions)})"
            
            selected_pos = free_positions[idx]
            assert len(selected_pos) == 2, f"selected_pos must have 2 elements, got {len(selected_pos)}"
            self.pos = tuple(int(x) for x in selected_pos)
        else:
            assert self.start_pos is not None, "start_pos validation failed"
            self.pos = tuple(self.start_pos)
        
        # Validate final position
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        assert 0 <= r < self.h and 0 <= c < self.w, f"pos {self.pos} out of bounds for {self.h}x{self.w} room"
        
        return self.get_observation()

    def step(self, action):
        # Strict input validation
        assert isinstance(action, (int, np.integer)), f"action must be int, got {type(action)}"
        assert 0 <= action < self.n_actions, f"action {action} out of range [0, {self.n_actions})"
        assert action in self.action_map, f"action {action} not in action_map {list(self.action_map.keys())}"
        
        # Validate current state
        assert hasattr(self, 'pos'), "pos must be set (call reset() first)"
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        
        dr, dc = self.action_map[action]
        assert isinstance(dr, int) and isinstance(dc, int), f"action_map values must be int, got {type(dr)}, {type(dc)}"
        
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        
        new_r, new_c = r + dr, c + dc

        # Boundary check
        if not (0 <= new_r < self.h and 0 <= new_c < self.w):
            return self.get_observation(), False
        
        # Wall check
        if self.room[new_r, new_c] == -1:
            return self.get_observation(), False

        # Invisible wall checks
        flat_idx = r * self.w + c
        assert isinstance(flat_idx, int), f"flat_idx must be int, got {type(flat_idx)}"
        assert 0 <= flat_idx < self.h * self.w, f"flat_idx {flat_idx} out of bounds for {self.h}x{self.w} room"
        
        if action == 0 and flat_idx in self.no_up:
            return self.get_observation(), False
        if action == 1 and flat_idx in self.no_down:
            return self.get_observation(), False
        if action == 2 and flat_idx in self.no_left:
            return self.get_observation(), False
        if action == 3 and flat_idx in self.no_right:
            return self.get_observation(), False

        # Update position
        self.pos = (int(new_r), int(new_c))
        return self.get_observation(), True

    def get_observation(self):
        # Strict validation for observation generation
        assert hasattr(self, 'pos'), "pos must be set (call reset() first)"
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        assert 0 <= r < self.h and 0 <= c < self.w, f"pos {self.pos} out of bounds for {self.h}x{self.w} room"
        
        # Check walls in all directions
        up    = int(self.room[r - 1, c] != -1) if r > 0     else 0
        down  = int(self.room[r + 1, c] != -1) if r < self.h - 1 else 0
        left  = int(self.room[r, c - 1] != -1) if c > 0     else 0
        right = int(self.room[r, c + 1] != -1) if c < self.w - 1 else 0
        
        # Validate wall check results
        for val, name in [(up, "up"), (down, "down"), (left, "left"), (right, "right")]:
            assert isinstance(val, int), f"{name} must be int, got {type(val)}"
            assert val in [0, 1], f"{name} must be 0 or 1, got {val}"
        
        obs = (up << 3) + (down << 2) + (left << 1) + right
        
        # Final validation
        assert isinstance(obs, int), f"obs must be int, got {type(obs)}"
        assert 0 <= obs <= 15, f"obs {obs} out of range [0, 15]"
        
        return obs

class RoomTorchAdapter(CSCGEnvironmentAdapter):
    def __init__(self, room_tensor, no_up=[], no_down=[], no_left=[], no_right=[], start_pos=None, seed=42):
        # Strict input validation for PyTorch tensor
        assert isinstance(room_tensor, torch.Tensor), f"room_tensor must be torch.Tensor, got {type(room_tensor)}"
        assert room_tensor.ndim == 2, f"room_tensor must be 2D, got {room_tensor.ndim}D"
        assert room_tensor.dtype in [torch.int32, torch.int64, torch.long], f"room_tensor must be integer type, got {room_tensor.dtype}"
        assert room_tensor.numel() > 0, "room_tensor cannot be empty"
        
        super().__init__(seed=seed)
        
        # Ensure room tensor is on the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.room = room_tensor.to(self.device)
        self.h, self.w = int(self.room.shape[0]), int(self.room.shape[1])
        
        # Validate dimensions
        assert self.h > 0 and self.w > 0, f"Invalid room dimensions: {self.h}x{self.w}"
        
        # Validate start position
        if start_pos is not None:
            assert isinstance(start_pos, (tuple, list)), f"start_pos must be tuple/list, got {type(start_pos)}"
            assert len(start_pos) == 2, f"start_pos must have 2 elements, got {len(start_pos)}"
            r, c = start_pos
            assert isinstance(r, int) and isinstance(c, int), f"start_pos elements must be int, got {type(r)}, {type(c)}"
            assert 0 <= r < self.h and 0 <= c < self.w, f"start_pos {start_pos} out of bounds for {self.h}x{self.w} room"
        
        self.start_pos = start_pos
        
        # Validate wall lists
        for wall_list, name in [(no_up, "no_up"), (no_down, "no_down"), (no_left, "no_left"), (no_right, "no_right")]:
            assert isinstance(wall_list, (list, tuple, set)), f"{name} must be list/tuple/set, got {type(wall_list)}"
            for idx in wall_list:
                assert isinstance(idx, int), f"{name} elements must be int, got {type(idx)}"
                assert 0 <= idx < self.h * self.w, f"{name} index {idx} out of bounds for {self.h}x{self.w} room"
        
        self.no_up = set(no_up)
        self.no_down = set(no_down)
        self.no_left = set(no_left)
        self.no_right = set(no_right)

        self.action_map = ACTIONS
        self.n_actions = 4
        
        # Post-initialization validation
        assert isinstance(self.n_actions, int), f"n_actions must be int, got {type(self.n_actions)}"
        assert self.n_actions == len(ACTIONS), f"n_actions mismatch: {self.n_actions} != {len(ACTIONS)}"
        assert self.room.device == self.device, f"Room tensor device mismatch: {self.room.device} != {self.device}"
        
        self.reset()

    def reset(self):
        # Strict validation for reset operation
        assert hasattr(self, 'room'), "room must be initialized"
        assert hasattr(self, 'rng'), "rng must be initialized"
        assert isinstance(self.room, torch.Tensor), f"room must be torch.Tensor, got {type(self.room)}"
        assert self.room.device == self.device, f"room device mismatch: {self.room.device} != {self.device}"
        
        if self.start_pos is None:
            free_positions = (self.room != -1).nonzero(as_tuple=False)
            assert isinstance(free_positions, torch.Tensor), f"free_positions must be torch.Tensor, got {type(free_positions)}"
            assert len(free_positions) > 0, "No free positions found in room"
            
            idx = self.rng.choice(len(free_positions))
            assert isinstance(idx, (int, np.integer)), f"idx must be int, got {type(idx)}"
            assert 0 <= idx < len(free_positions), f"idx {idx} out of range [0, {len(free_positions)})"
            
            selected_pos = free_positions[idx].tolist()
            assert isinstance(selected_pos, list), f"selected_pos must be list, got {type(selected_pos)}"
            assert len(selected_pos) == 2, f"selected_pos must have 2 elements, got {len(selected_pos)}"
            self.pos = tuple(int(x) for x in selected_pos)
        else:
            assert self.start_pos is not None, "start_pos validation failed"
            self.pos = tuple(self.start_pos)
        
        # Validate final position
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        assert 0 <= r < self.h and 0 <= c < self.w, f"pos {self.pos} out of bounds for {self.h}x{self.w} room"
        
        return self.get_observation()

    def step(self, action):
        # Strict input validation
        assert isinstance(action, (int, np.integer)), f"action must be int, got {type(action)}"
        assert 0 <= action < self.n_actions, f"action {action} out of range [0, {self.n_actions})"
        assert action in self.action_map, f"action {action} not in action_map {list(self.action_map.keys())}"
        
        # Validate current state
        assert hasattr(self, 'pos'), "pos must be set (call reset() first)"
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        assert hasattr(self, 'room'), "room must be initialized"
        assert self.room.device == self.device, f"room device mismatch: {self.room.device} != {self.device}"
        
        dr, dc = self.action_map[action]
        assert isinstance(dr, int) and isinstance(dc, int), f"action_map values must be int, got {type(dr)}, {type(dc)}"
        
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        
        new_r, new_c = r + dr, c + dc

        # Boundary check
        if not (0 <= new_r < self.h and 0 <= new_c < self.w):
            return self.get_observation(), False
        
        # Wall check (tensor comparison)
        wall_check = self.room[new_r, new_c] == -1
        assert isinstance(wall_check, torch.Tensor), f"wall_check must be tensor, got {type(wall_check)}"
        if wall_check.item():
            return self.get_observation(), False

        # Invisible wall checks
        flat_idx = r * self.w + c
        assert isinstance(flat_idx, int), f"flat_idx must be int, got {type(flat_idx)}"
        assert 0 <= flat_idx < self.h * self.w, f"flat_idx {flat_idx} out of bounds for {self.h}x{self.w} room"
        
        if action == 0 and flat_idx in self.no_up:
            return self.get_observation(), False
        if action == 1 and flat_idx in self.no_down:
            return self.get_observation(), False
        if action == 2 and flat_idx in self.no_left:
            return self.get_observation(), False
        if action == 3 and flat_idx in self.no_right:
            return self.get_observation(), False

        # Update position
        self.pos = (int(new_r), int(new_c))
        return self.get_observation(), True

    def get_observation(self):
        # Strict validation for observation generation
        assert hasattr(self, 'pos'), "pos must be set (call reset() first)"
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        assert hasattr(self, 'room'), "room must be initialized"
        assert self.room.device == self.device, f"room device mismatch: {self.room.device} != {self.device}"
        
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        assert 0 <= r < self.h and 0 <= c < self.w, f"pos {self.pos} out of bounds for {self.h}x{self.w} room"
        
        # Check walls in all directions (convert tensor comparisons to Python ints)
        up    = int((self.room[r - 1, c] != -1).item()) if r > 0     else 0
        down  = int((self.room[r + 1, c] != -1).item()) if r < self.h - 1 else 0
        left  = int((self.room[r, c - 1] != -1).item()) if c > 0     else 0
        right = int((self.room[r, c + 1] != -1).item()) if c < self.w - 1 else 0
        
        # Validate wall check results
        for val, name in [(up, "up"), (down, "down"), (left, "left"), (right, "right")]:
            assert isinstance(val, int), f"{name} must be int, got {type(val)}"
            assert val in [0, 1], f"{name} must be 0 or 1, got {val}"
        
        obs = (up << 3) + (down << 2) + (left << 1) + right
        
        # Final validation
        assert isinstance(obs, int), f"obs must be int, got {type(obs)}"
        assert 0 <= obs <= 15, f"obs {obs} out of range [0, 15]"
        
        return obs
    
def save_room_plot(room, save_path_base, cmap='viridis', title=None, show_grid=True, 
                  save_formats=['pdf', 'png'], figsize=(10, 8)):
    """
    Save room plot with enhanced visualization options.
    
    Args:
        room (array-like): Room layout data
        save_path_base (str): Base path for saving (without extension)
        cmap (str or colormap): Colormap to use for plotting
        title (str, optional): Custom title for the plot
        show_grid (bool): Whether to show grid lines
        save_formats (list): List of formats to save ['pdf', 'png', 'svg']
        figsize (tuple): Figure size as (width, height)
    """
    # Input validation
    assert room is not None, "room cannot be None"
    assert isinstance(save_path_base, str), f"save_path_base must be str, got {type(save_path_base)}"
    assert len(save_path_base) > 0, "save_path_base cannot be empty"
    assert isinstance(save_formats, (list, tuple)), f"save_formats must be list/tuple, got {type(save_formats)}"
    
    # Convert to numpy if needed
    if isinstance(room, torch.Tensor):
        if room.is_cuda:
            room = room.cpu()
        room = room.numpy()
    
    room = np.asarray(room)
    assert room.ndim == 2, f"room must be 2D, got {room.ndim}D"
    assert room.shape[0] > 0 and room.shape[1] > 0, f"room must have positive dimensions, got {room.shape}"
    
    # Create enhanced plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine appropriate colormap based on data
    unique_vals = np.unique(room)
    if len(unique_vals) <= 10:
        # Use discrete colormap for few unique values
        im = ax.imshow(room, cmap=cmap, interpolation='nearest')
    else:
        # Use continuous colormap for many values
        im = ax.imshow(room, cmap=cmap)
    
    # Enhanced colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    if -1 in unique_vals and len(unique_vals) <= 5:
        # Likely a maze with walls (-1) and rooms
        cbar.set_label('Cell Type (-1: Wall, ≥0: Room)', rotation=270, labelpad=15)
    else:
        cbar.set_label('Room Value', rotation=270, labelpad=15)
    
    # Set title
    if title is None:
        title = f"Room Layout ({room.shape[0]}×{room.shape[1]})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Axis labels
    ax.set_xlabel("Column Index", fontsize=12)
    ax.set_ylabel("Row Index", fontsize=12)
    
    # Grid options
    if show_grid:
        ax.grid(True, alpha=0.3, linewidth=0.5, color='white')
        
    # Add room statistics as text
    stats_text = f"Size: {room.shape[0]}×{room.shape[1]}\nUnique values: {len(unique_vals)}"
    if -1 in unique_vals:
        n_walls = np.sum(room == -1)
        n_free = room.size - n_walls
        stats_text += f"\nWalls: {n_walls}\nFree: {n_free}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save in specified formats
    saved_files = []
    for fmt in save_formats:
        if fmt.lower() in ['pdf', 'png', 'svg', 'eps']:
            save_path = f"{save_path_base}.{fmt.lower()}"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            saved_files.append(save_path)
    
    plt.close()
    
    if saved_files:
        print(f"Room plot saved to: {', '.join(saved_files)}")
    else:
        print("Warning: No valid save formats specified")
