"""
Room Environment Utilities

Helper functions for working with room-based environments and CHMM models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_room_adapter(room_data, **kwargs):
    """
    Create a room adapter from room data.
    
    Args:
        room_data: Room layout (torch.Tensor or numpy.ndarray)
                  Should be 2D array where -1 represents walls
        **kwargs: Additional arguments for adapter (e.g., seed)
        
    Returns:
        RoomAdapter: Configured room adapter
    """
    # Strict input validation
    assert room_data is not None, "room_data cannot be None"
    assert isinstance(room_data, (torch.Tensor, np.ndarray)), f"room_data must be torch.Tensor or numpy.ndarray, got {type(room_data)}"
    
    # Validate room_data shape and content
    if hasattr(room_data, 'ndim'):
        assert room_data.ndim == 2, f"room_data must be 2D, got {room_data.ndim}D"
        assert room_data.shape[0] > 0 and room_data.shape[1] > 0, f"room_data must have positive dimensions, got {room_data.shape}"
    
    # Convert to numpy array if needed
    if isinstance(room_data, torch.Tensor):
        if room_data.is_cuda or room_data.device.type == 'mps':
            room_data = room_data.cpu()
        room_data = room_data.numpy()
    
    # Validate converted array
    assert isinstance(room_data, np.ndarray), f"room_data must be numpy array after conversion, got {type(room_data)}"
    
    # Import RoomAdapter
    try:
        from .room_adapter import RoomAdapter
    except ImportError:
        from room_adapter import RoomAdapter
    
    # Create adapter
    adapter = RoomAdapter(room_data, **kwargs)
    
    # Final validation
    assert hasattr(adapter, 'reset'), "adapter must have reset method"
    assert hasattr(adapter, 'step'), "adapter must have step method"
    assert hasattr(adapter, 'get_observation'), "adapter must have get_observation method"
    assert hasattr(adapter, 'check_avail_actions'), "adapter must have check_avail_actions method"
    
    return adapter

def get_room_n_clones(n_clones_per_obs=1, device=None):
    """
    Get n_clones tensor for room navigation CHMM model.
    
    Args:
        n_clones_per_obs (int): Number of clones per observation type
        device: PyTorch device (auto-detected if None)
        
    Returns:
        torch.Tensor: n_clones tensor for model initialization
    """
    # Strict input validation
    assert isinstance(n_clones_per_obs, int), f"n_clones_per_obs must be int, got {type(n_clones_per_obs)}"
    assert n_clones_per_obs > 0, f"n_clones_per_obs must be positive, got {n_clones_per_obs}"
    assert n_clones_per_obs <= 1000, f"n_clones_per_obs too large (max 1000), got {n_clones_per_obs}"
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
    
    # Validate device
    assert isinstance(device, torch.device), f"device must be torch.device, got {type(device)}"
    
    # For room navigation: 4-bit observation (up, down, left, right availability)
    n_obs_types = 16  # 2^4 possible combinations
    assert isinstance(n_obs_types, int), f"n_obs_types must be int, got {type(n_obs_types)}"
    assert n_obs_types == 16, f"n_obs_types must be 16 for room navigation, got {n_obs_types}"
    
    result = torch.full((n_obs_types,), n_clones_per_obs, dtype=torch.int64, device=device)
    
    # Final validation
    assert isinstance(result, torch.Tensor), f"result must be torch.Tensor, got {type(result)}"
    assert result.shape == (n_obs_types,), f"result shape must be ({n_obs_types},), got {result.shape}"
    assert result.dtype == torch.int64, f"result dtype must be int64, got {result.dtype}"
    assert result.device.type == device.type, f"result device type mismatch: {result.device} != {device}"
    assert torch.all(result == n_clones_per_obs), f"all result values must equal {n_clones_per_obs}"
    
    return result
