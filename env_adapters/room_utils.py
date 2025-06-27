"""
Room Environment Utilities

Helper functions for working with room-based environments and CHMM models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    # Strict input validation
    assert room_data is not None, "room_data cannot be None"
    assert isinstance(room_data, (torch.Tensor, np.ndarray)), f"room_data must be torch.Tensor or numpy.ndarray, got {type(room_data)}"
    assert isinstance(adapter_type, str), f"adapter_type must be str, got {type(adapter_type)}"
    assert adapter_type in ["torch", "numpy"], f"adapter_type must be 'torch' or 'numpy', got '{adapter_type}'"
    
    # Validate room_data shape and content
    if hasattr(room_data, 'ndim'):
        assert room_data.ndim == 2, f"room_data must be 2D, got {room_data.ndim}D"
        assert room_data.shape[0] > 0 and room_data.shape[1] > 0, f"room_data must have positive dimensions, got {room_data.shape}"
    
    # Dynamic import to avoid circular import issues
    try:
        from .room_adapter import RoomTorchAdapter, RoomNPAdapter
    except ImportError:
        try:
            from room_adapter import RoomTorchAdapter, RoomNPAdapter
        except ImportError:
            raise ImportError("Could not import room adapters. Use colab_imports.py instead.")
    
    if adapter_type == "torch":
        if not isinstance(room_data, torch.Tensor):
            assert isinstance(room_data, np.ndarray), f"room_data conversion: expected numpy array, got {type(room_data)}"
            room_data = torch.tensor(room_data)
        
        # Validate converted tensor
        assert isinstance(room_data, torch.Tensor), f"room_data must be tensor after conversion, got {type(room_data)}"
        assert room_data.dtype in [torch.int32, torch.int64, torch.long], f"room_data must have integer dtype, got {room_data.dtype}"
        
        # Remove device from kwargs since RoomTorchAdapter handles device internally
        kwargs.pop('device', None)
        
        adapter = RoomTorchAdapter(room_data, **kwargs)
    else:
        if isinstance(room_data, torch.Tensor):
            # Safe tensor to numpy conversion
            if room_data.is_cuda:
                room_data = room_data.cpu()
            room_data = room_data.numpy()
        
        # Validate converted array
        assert isinstance(room_data, np.ndarray), f"room_data must be numpy array after conversion, got {type(room_data)}"
        assert room_data.dtype in [np.int32, np.int64], f"room_data must have integer dtype, got {room_data.dtype}"
        
        adapter = RoomNPAdapter(room_data, **kwargs)
    
    # Final validation
    assert hasattr(adapter, 'reset'), "adapter must have reset method"
    assert hasattr(adapter, 'step'), "adapter must have step method"
    assert hasattr(adapter, 'get_observation'), "adapter must have get_observation method"
    
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    assert result.device == device, f"result device mismatch: {result.device} != {device}"
    assert torch.all(result == n_clones_per_obs), f"all result values must equal {n_clones_per_obs}"
    
    return result


def demo_room_setup():
    """
    Create a demo room setup for testing.
    
    Returns:
        tuple: (room_adapter, n_clones, sample_data)
    """
    # Create a simple 5x5 room with walls
    room_tensor = torch.randint(0, 2, size=[5, 5], dtype=torch.long)
    
    # Validate initial room tensor
    assert isinstance(room_tensor, torch.Tensor), f"room_tensor must be torch.Tensor, got {type(room_tensor)}"
    assert room_tensor.shape == (5, 5), f"room_tensor shape must be (5, 5), got {room_tensor.shape}"
    assert room_tensor.dtype == torch.long, f"room_tensor dtype must be long, got {room_tensor.dtype}"
    
    # Add walls
    room_tensor[0, :] = -1  # Top wall
    room_tensor[-1, :] = -1  # Bottom wall
    room_tensor[:, 0] = -1  # Left wall
    room_tensor[:, -1] = -1  # Right wall
    
    # Validate room has walls and open spaces
    assert torch.any(room_tensor == -1), "room_tensor must have walls (-1 values)"
    assert torch.any(room_tensor != -1), "room_tensor must have open spaces (non -1 values)"
    
    # Create adapter
    adapter = create_room_adapter(room_tensor)
    assert hasattr(adapter, 'reset'), "adapter must have reset method"
    assert hasattr(adapter, 'generate_sequence'), "adapter must have generate_sequence method"
    
    # Create n_clones
    n_clones = get_room_n_clones(n_clones_per_obs=2)
    assert isinstance(n_clones, torch.Tensor), f"n_clones must be torch.Tensor, got {type(n_clones)}"
    assert n_clones.shape == (16,), f"n_clones shape must be (16,), got {n_clones.shape}"
    
    # Generate sample data
    x_seq, a_seq = adapter.generate_sequence(100)
    
    # Validate sample data
    assert isinstance(x_seq, np.ndarray), f"x_seq must be numpy array, got {type(x_seq)}"
    assert isinstance(a_seq, np.ndarray), f"a_seq must be numpy array, got {type(a_seq)}"
    assert len(x_seq) == len(a_seq), f"sequence lengths must match: x={len(x_seq)}, a={len(a_seq)}"
    assert len(x_seq) <= 100, f"sequence length must not exceed 100, got {len(x_seq)}"
    assert len(x_seq) > 0, "sequence must not be empty"
    
    # Validate observation and action ranges
    assert np.all(x_seq >= 0) and np.all(x_seq <= 15), f"observations must be in [0, 15], got range [{x_seq.min()}, {x_seq.max()}]"
    assert np.all(a_seq >= 0) and np.all(a_seq <= 3), f"actions must be in [0, 3], got range [{a_seq.min()}, {a_seq.max()}]"
    
    sample_data = (x_seq, a_seq)
    
    return adapter, n_clones, sample_data


def get_obs_colormap(n_obs):
    """
    Get a colormap for observations.
    
    Args:
        n_obs (int): Number of observation types
        
    Returns:
        matplotlib colormap
    """
    assert isinstance(n_obs, int), f"n_obs must be int, got {type(n_obs)}"
    assert n_obs > 0, f"n_obs must be positive, got {n_obs}"
    assert n_obs <= 100, f"n_obs too large (max 100), got {n_obs}"
    
    cmap = cm.get_cmap('tab20', n_obs)
    return cmap


def clone_to_obs_map(n_clones):
    """
    Create mapping from clone indices to observation indices.
    
    Args:
        n_clones (array-like): Number of clones per observation
        
    Returns:
        dict: Mapping from clone index to observation index
    """
    # Convert to numpy if needed for consistent handling
    if isinstance(n_clones, torch.Tensor):
        n_clones = n_clones.cpu().numpy()
    
    assert isinstance(n_clones, (np.ndarray, list, tuple)), f"n_clones must be array-like, got {type(n_clones)}"
    n_clones = np.asarray(n_clones)
    assert n_clones.ndim == 1, f"n_clones must be 1D, got {n_clones.ndim}D"
    assert len(n_clones) > 0, "n_clones cannot be empty"
    assert np.all(n_clones > 0), "all n_clones values must be positive"
    
    mapping = {}
    idx = 0
    for obs_id, n in enumerate(n_clones):
        for _ in range(int(n)):
            mapping[idx] = obs_id
            idx += 1
    
    # Validate mapping
    assert len(mapping) == np.sum(n_clones), f"mapping length mismatch: {len(mapping)} != {np.sum(n_clones)}"
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in mapping.items()), "mapping must have int keys and values"
    
    return mapping


def top_k_used_clones(states, k):
    """
    Get top k most frequently used clones.
    
    Args:
        states (array-like): State sequence
        k (int): Number of top clones to return
        
    Returns:
        list: List of (clone_id, count) tuples sorted by frequency
    """
    assert isinstance(k, int), f"k must be int, got {type(k)}"
    assert k > 0, f"k must be positive, got {k}"
    
    # Convert to numpy if needed
    if isinstance(states, torch.Tensor):
        states = states.cpu().numpy()
    
    assert isinstance(states, (np.ndarray, list, tuple)), f"states must be array-like, got {type(states)}"
    states = np.asarray(states)
    assert states.ndim == 1, f"states must be 1D, got {states.ndim}D"
    assert len(states) > 0, "states cannot be empty"
    
    unique, counts = np.unique(states, return_counts=True)
    freq_dict = dict(zip(unique, counts))
    
    # Validate frequency dict
    assert len(freq_dict) > 0, "freq_dict cannot be empty"
    assert all(isinstance(k, (int, np.integer)) and isinstance(v, (int, np.integer)) for k, v in freq_dict.items()), "freq_dict must have int keys and values"
    
    result = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)[:k]
    
    # Validate result
    assert len(result) <= k, f"result length {len(result)} exceeds k {k}"
    assert len(result) <= len(freq_dict), f"result length {len(result)} exceeds unique states {len(freq_dict)}"
    
    return result


def count_used_clones(chmm, x, a):
    """
    Count number of unique clones used per observation type.
    
    Args:
        chmm: CHMM model
        x (array-like): Observation sequence
        a (array-like): Action sequence
        
    Returns:
        dict: Mapping from observation_id to number of unique clones used
    """
    # Validate inputs
    assert hasattr(chmm, 'decode'), "chmm must have decode method"
    assert hasattr(chmm, 'n_clones'), "chmm must have n_clones attribute"
    
    # Convert sequences to appropriate format for decoding
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    
    # Decode to get state sequence
    _, states = chmm.decode(x, a)
    
    # Convert to numpy for processing
    if isinstance(states, torch.Tensor):
        states = states.cpu().numpy()
    
    used_clones = np.unique(states)
    
    # Get clone ranges for each observation type
    if isinstance(chmm.n_clones, torch.Tensor):
        n_clones = chmm.n_clones.cpu().numpy()
    else:
        n_clones = np.asarray(chmm.n_clones)
    
    clone_ranges = np.cumsum(np.concatenate([[0], n_clones]))
    counts = {}

    for obs_id in range(len(n_clones)):
        start = clone_ranges[obs_id]
        end = clone_ranges[obs_id + 1]
        clones_in_use = [s for s in used_clones if start <= s < end]
        counts[obs_id] = len(clones_in_use)

    # Validate counts
    assert len(counts) == len(n_clones), f"counts length {len(counts)} != n_clones length {len(n_clones)}"
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in counts.items()), "counts must have int keys and values"
    assert all(v >= 0 for v in counts.values()), "all counts must be non-negative"

    return counts