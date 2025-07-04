"""
2D Agent Adapter

Agent for traversing 2D environments like rooms and mazes.
"""

import numpy as np
import torch
from typing import Tuple, List
from .base_agent import BaseAgent

# Global debug flag - will be set by the environment adapter
DEBUG = False

def set_debug_mode(debug: bool):
    """Set global debug mode for assertions"""
    global DEBUG
    DEBUG = debug

class Agent2D(BaseAgent):
    """
    Agent class for traversing 2D environments and generating training sequences.
    
    Inherits from BaseAgent and implements traverse method for 2D environments.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize 2D agent with random seed.
        
        Args:
            seed: Random seed for reproducible behavior
        """
        super().__init__(seed)
    
    def traverse(self, environment, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Traverse environment for specified sequence length.
        
        OPTIMIZED: Runs trajectory generation on CPU for speed, then transfers to target device.
        
        Args:
            environment: Environment to traverse (must have check_avail_actions method)
            seq_len: Number of steps to take
            
        Returns:
            Tuple of:
            - observations: torch.Tensor of shape (seq_len,) with dtype torch.int64
            - actions: torch.Tensor of shape (seq_len,) with dtype torch.int64  
            - path: List of (row, col) coordinates showing traversed path
        """
        if DEBUG:
            assert hasattr(environment, 'check_avail_actions'), "Environment must have check_avail_actions method"
            assert hasattr(environment, 'get_observation'), "Environment must have get_observation method"
            assert hasattr(environment, 'step'), "Environment must have step method"
            assert hasattr(environment, 'reset'), "Environment must have reset method"
            assert hasattr(environment, 'device'), "Environment must have device attribute"
            assert isinstance(seq_len, int) and seq_len > 0, f"seq_len must be positive int, got {seq_len}"
        
        target_device = environment.device
        
        # OPTIMIZATION: Generate trajectory on CPU using numpy for speed
        # Get room data from environment for direct CPU processing
        room_cpu = environment.room.cpu().numpy()
        height, width = room_cpu.shape
        
        # Pre-allocate CPU arrays
        observations_np = np.zeros(seq_len, dtype=np.int64)
        actions_np = np.zeros(seq_len, dtype=np.int64)
        path = []
        
        # Random starting position (same logic as environment.reset())
        non_wall_positions = np.where(room_cpu >= 0)
        if len(non_wall_positions[0]) == 0:
            raise ValueError("No valid positions found in room")
        
        idx = self.rng.randint(0, len(non_wall_positions[0]))
        row = int(non_wall_positions[0][idx])
        col = int(non_wall_positions[1][idx])
        
        # Action mappings: 0=up, 1=down, 2=left, 3=right
        action_deltas = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        
        # Generate trajectory using fast CPU operations
        for step in range(seq_len):
            # Record current observation
            observations_np[step] = room_cpu[row, col]
            path.append((row, col))
            
            # Get available actions (same logic as environment.check_avail_actions)
            available_actions = []
            for action, (dr, dc) in action_deltas.items():
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < height and 
                    0 <= new_col < width and 
                    room_cpu[new_row, new_col] >= 0):
                    available_actions.append(action)
            
            if DEBUG:
                assert len(available_actions) > 0, f"No available actions at position ({row}, {col})"
            
            # Randomly select action
            action = int(self.rng.choice(available_actions))
            actions_np[step] = action
            
            # Take action
            dr, dc = action_deltas[action]
            row, col = row + dr, col + dc
            
            if DEBUG:
                assert 0 <= row < height, f"Row {row} out of bounds"
                assert 0 <= col < width, f"Col {col} out of bounds"
                assert room_cpu[row, col] >= 0, f"Moved to invalid position ({row}, {col})"
        
        # Convert to torch tensors on target device
        observations = torch.from_numpy(observations_np).to(target_device)
        actions = torch.from_numpy(actions_np).to(target_device)
        
        if DEBUG:
            assert len(path) == seq_len, f"Path length {len(path)} != seq_len {seq_len}"
            assert observations.shape == (seq_len,), f"observations shape {observations.shape} != ({seq_len},)"
            assert actions.shape == (seq_len,), f"actions shape {actions.shape} != ({seq_len},)"
            assert observations.device == target_device, f"observations on wrong device: {observations.device} != {target_device}"
            assert actions.device == target_device, f"actions on wrong device: {actions.device} != {target_device}"
        
        return observations, actions, path