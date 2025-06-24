"""
Room Environment Utilities

Helper functions for working with room-based environments and CHMM models.
"""

import torch


def create_room_adapter(room_data, adapter_type="torch", **kwargs):
    """
    Create a room adapter from room data.
    
    Args:
        room_data: Room layout (torch.Tensor or numpy.ndarray)
        adapter_type (str): "torch" or "numpy"
        **kwargs: Additional arguments for adapter
        
    Returns:
        CSCGEnvironmentAdapter: Configured room adapter
    """
    # Dynamic import to avoid circular import issues
    try:
        from .room_adapter import RoomTorchAdapter, RoomNPAdapter
    except ImportError:
        try:
            from room_adapter import RoomTorchAdapter, RoomNPAdapter
        except ImportError:
            raise ImportError("Could not import room adapters. Make sure room_adapter.py is available.")
    
    if adapter_type == "torch":
        if not isinstance(room_data, torch.Tensor):
            room_data = torch.tensor(room_data)
        return RoomTorchAdapter(room_data, **kwargs)
    else:
        if isinstance(room_data, torch.Tensor):
            room_data = room_data.cpu().numpy()
        return RoomNPAdapter(room_data, **kwargs)


def get_room_n_clones(n_clones_per_obs=1, device=None):
    """
    Get n_clones tensor for room navigation CHMM model.
    
    Args:
        n_clones_per_obs (int): Number of clones per observation type
        device: PyTorch device (auto-detected if None)
        
    Returns:
        torch.Tensor: n_clones tensor for model initialization
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # For room navigation: 4-bit observation (up, down, left, right availability)
    n_obs_types = 16  # 2^4 possible combinations
    return torch.full((n_obs_types,), n_clones_per_obs, dtype=torch.int64, device=device)


def demo_room_setup():
    """
    Create a demo room setup for testing.
    
    Returns:
        tuple: (room_adapter, n_clones, sample_data)
    """
    # Create a simple 5x5 room with walls
    room_tensor = torch.randint(0, 2, size=[5, 5], dtype=torch.long)
    room_tensor[0, :] = -1  # Top wall
    room_tensor[-1, :] = -1  # Bottom wall
    room_tensor[:, 0] = -1  # Left wall
    room_tensor[:, -1] = -1  # Right wall
    
    # Create adapter using dynamic import
    adapter = create_room_adapter(room_tensor)
    n_clones = get_room_n_clones(n_clones_per_obs=2)
    
    # Generate sample data
    x_seq, a_seq = adapter.generate_sequence(100)
    
    return adapter, n_clones, (x_seq, a_seq)