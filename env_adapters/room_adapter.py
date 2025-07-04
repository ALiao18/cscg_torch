import argparse
import numpy as np
import torch
import matplotlib.cm as cm
from typing import Tuple, List
from .base_adapter import CSCGEnvironmentAdapter

# Action mappings: 0=up, 1=down, 2=left, 3=right
ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

# Global debug flag
DEBUG = False

def set_debug_mode(debug: bool):
    """Set global debug mode for assertions"""
    global DEBUG
    DEBUG = debug

class RoomAdapter(CSCGEnvironmentAdapter):
    """
    Room environment adapter for rooms with walls.
    
    Takes a room as numpy array where:
    - Each cell contains an observation value (>= 0)
    - Walls are marked as -1
    - Agent can only move to non-wall cells
    """
    
    def __init__(self, room_array: np.ndarray, seed: int = 42):
        """
        Initialize room adapter.
        
        Args:
            room_array: 2D numpy array representing the room layout
                       Values >= 0 are valid cells with observation values
                       Value -1 represents walls
            seed: Random seed (fixed at 42)
        """
        if DEBUG:
            assert isinstance(room_array, np.ndarray), f"room_array must be numpy array, got {type(room_array)}"
            assert room_array.ndim == 2, f"room_array must be 2D, got {room_array.ndim}D"
            assert room_array.size > 0, "room_array cannot be empty"
            assert isinstance(seed, int), f"seed must be int, got {type(seed)}"
            # Check that there are non-wall cells
            assert np.any(room_array >= 0), "room_array must contain at least one non-wall cell (value >= 0)"
        
        super().__init__(seed=seed)
        
        self.room_height, self.room_width = room_array.shape
        self.n_actions = 4
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps:0")
        else:
            self.device = torch.device("cpu")
        
        # Convert room to tensor and move to device
        self.room = torch.from_numpy(room_array).to(self.device, dtype=torch.int64)
        
        # Initialize position
        self.pos = None
        self.reset()
    
    def reset(self):
        """Reset agent to random position in room (non-wall cell)"""
        # Find all non-wall positions
        non_wall_positions = torch.where(self.room >= 0)
        
        if DEBUG:
            assert len(non_wall_positions[0]) > 0, "No non-wall positions found in room"
        
        # Randomly select a non-wall position
        idx = torch.randint(0, len(non_wall_positions[0]), (1,), device=self.device).item()
        row = non_wall_positions[0][idx].item()
        col = non_wall_positions[1][idx].item()
        
        self.pos = (row, col)
        
        return self.get_observation()
    
    def get_observation(self) -> int:
        """Get observation at current position"""
        if DEBUG:
            assert self.pos is not None, "Position not set, call reset() first"
        
        row, col = self.pos
        obs = self.room[row, col].item()
        
        if DEBUG:
            assert isinstance(obs, int), f"obs must be int, got {type(obs)}"
            assert obs >= 0, f"obs must be non-negative (not a wall), got {obs}"
        
        return obs
    
    def check_avail_actions(self, pos: Tuple[int, int]) -> List[int]:
        """
        Check available actions at given position.
        
        Args:
            pos: (row, col) position in room
            
        Returns:
            List of available action indices
        """
        if DEBUG:
            assert isinstance(pos, tuple) and len(pos) == 2, f"pos must be (row, col) tuple, got {pos}"
        
        row, col = pos
        
        if DEBUG:
            assert 0 <= row < self.room_height, f"row {row} out of bounds [0, {self.room_height})"
            assert 0 <= col < self.room_width, f"col {col} out of bounds [0, {self.room_width})"
            # Check that current position is not a wall
            current_obs = self.room[row, col].item()
            assert current_obs >= 0, f"Current position {pos} is a wall (obs={current_obs})"
        
        available_actions = []
        
        for action, (dr, dc) in ACTIONS.items():
            new_row, new_col = row + dr, col + dc
            
            # Check if new position is within bounds and not a wall
            if (0 <= new_row < self.room_height and 
                0 <= new_col < self.room_width and 
                self.room[new_row, new_col].item() >= 0):
                available_actions.append(action)
        
        if DEBUG:
            assert len(available_actions) > 0, f"No available actions at position {pos}"
        
        return available_actions
    
    def step(self, action: int) -> Tuple[int, bool]:
        """
        Take action and return new observation and validity.
        
        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            Tuple of (new_observation, action_was_valid)
        """
        if DEBUG:
            assert isinstance(action, int), f"action must be int, got {type(action)}"
            assert 0 <= action < self.n_actions, f"action {action} out of range [0, {self.n_actions})"
            assert self.pos is not None, "Position not set, call reset() first"
        
        # Check if action is valid from current position
        available_actions = self.check_avail_actions(self.pos)
        
        if action not in available_actions:
            return self.get_observation(), False
        
        # Execute action
        dr, dc = ACTIONS[action]
        row, col = self.pos
        new_row, new_col = row + dr, col + dc
        
        # Update position
        self.pos = (new_row, new_col)
        
        return self.get_observation(), True
    
    def get_obs_colormap(self, n_obs: int = None):
        """
        Get a colormap for observations.
        
        Args:
            n_obs (int, optional): Number of observation types. 
                                 If None, infers from room data.
            
        Returns:
            matplotlib colormap
        """
        if n_obs is None:
            # Infer from room data (excluding walls marked as -1)
            valid_obs = self.room[self.room >= 0]
            if len(valid_obs) > 0:
                n_obs = int(valid_obs.max().item()) + 1
            else:
                n_obs = 1  # fallback
        
        if DEBUG:
            assert isinstance(n_obs, int), f"n_obs must be int, got {type(n_obs)}"
            assert n_obs > 0, f"n_obs must be positive, got {n_obs}"
            assert n_obs <= 100, f"n_obs too large (max 100), got {n_obs}"
        
        cmap = cm.get_cmap('tab20', n_obs)
        return cmap


def main():
    """Main function to test the room adapter with debug mode"""
    # Import agent here to avoid circular imports
    from ..agent_adapters.agent_2d import Agent2D, set_debug_mode as set_agent_debug_mode
    
    parser = argparse.ArgumentParser(description="Room Adapter Test")
    parser.add_argument("--debug", action="store_true", default=False, 
                       help="Enable debug mode with assertions")
    parser.add_argument("--room_height", type=int, default=12, 
                       help="Height of the room (including walls)")
    parser.add_argument("--room_width", type=int, default=12, 
                       help="Width of the room (including walls)")
    parser.add_argument("--n_obs", type=int, default=8, 
                       help="Number of possible observations")
    parser.add_argument("--seq_len", type=int, default=100, 
                       help="Sequence length to generate")
    
    args = parser.parse_args()
    
    # Set debug mode for both environment and agent
    set_debug_mode(args.debug)
    set_agent_debug_mode(args.debug)
    
    # Create a sample room with walls for testing
    room_array = np.full((args.room_height, args.room_width), -1)  # Start with all walls
    
    # Create inner non-wall area
    inner_height = args.room_height - 2
    inner_width = args.room_width - 2
    inner_area = np.random.randint(0, args.n_obs, (inner_height, inner_width))
    room_array[1:-1, 1:-1] = inner_area
    
    # Create environment
    print(f"Creating {args.room_height}x{args.room_width} room with walls (-1) and observations 0-{args.n_obs-1}")
    print(f"Inner area: {inner_height}x{inner_width}")
    env = RoomAdapter(room_array, seed=42)
    
    # Create agent
    agent = Agent2D(seed=42)
    
    # Generate sequence
    print(f"Generating sequence of length {args.seq_len}")
    observations, actions, path = agent.traverse(env, args.seq_len)
    
    # Print results
    print(f"Generated sequence on device: {observations.device}")
    print(f"Observations shape: {observations.shape}, dtype: {observations.dtype}")
    print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"Path length: {len(path)}")
    
    # Show first few steps
    print("\nFirst 10 steps:")
    for i in range(min(10, args.seq_len)):
        print(f"Step {i}: pos={path[i]}, obs={observations[i].item()}, action={actions[i].item()}")


if __name__ == "__main__":
    main()