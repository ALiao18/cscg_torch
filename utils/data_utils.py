"""
Data utility functions for loading and saving room data.
"""

import os
import numpy as np
import torch
from typing import Union, Optional, Dict, Any

def load_room_data(room_name: str, 
                   data_dir: Optional[str] = None,
                   format: str = "auto") -> Union[np.ndarray, torch.Tensor]:
    """
    Load room data from various formats.
    
    Args:
        room_name: Name of the room (e.g., "room_20x20", "room_5x5")
        data_dir: Directory containing room files (defaults to package rooms/)
        format: File format ("npy", "pt", "txt", "auto")
        
    Returns:
        Room data as numpy array or torch tensor
        
    Examples:
        >>> room_data = load_room_data("room_20x20")
        >>> room_tensor = load_room_data("room_5x5", format="pt")
    """
    if data_dir is None:
        # Default to package rooms directory
        package_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(package_dir, "rooms")
    
    # Determine file format
    if format == "auto":
        # Try different extensions in order of preference
        for ext in ["npy", "pt", "txt"]:
            filepath = os.path.join(data_dir, f"{room_name}_16states.{ext}")
            if os.path.exists(filepath):
                format = ext
                break
        else:
            raise FileNotFoundError(f"No room data found for '{room_name}' in {data_dir}")
    else:
        filepath = os.path.join(data_dir, f"{room_name}_16states.{format}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Room file not found: {filepath}")
    
    # Load based on format
    if format == "npy":
        return np.load(filepath)
    elif format == "pt":
        return torch.load(filepath, map_location='cpu')
    elif format == "txt":
        return np.loadtxt(filepath, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_room_data(room_data: Union[np.ndarray, torch.Tensor],
                   room_name: str,
                   data_dir: Optional[str] = None,
                   formats: list = ["npy", "pt", "txt"]) -> Dict[str, str]:
    """
    Save room data in multiple formats.
    
    Args:
        room_data: Room layout data
        room_name: Name for the room
        data_dir: Directory to save files (defaults to package rooms/)
        formats: List of formats to save ("npy", "pt", "txt")
        
    Returns:
        Dictionary mapping format to saved filepath
        
    Examples:
        >>> paths = save_room_data(room_array, "custom_room")
        >>> print(paths["npy"])  # Path to numpy file
    """
    if data_dir is None:
        package_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(package_dir, "rooms")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Convert to numpy for processing
    if isinstance(room_data, torch.Tensor):
        room_np = room_data.cpu().numpy()
        room_tensor = room_data
    else:
        room_np = np.asarray(room_data)
        room_tensor = torch.tensor(room_data)
    
    saved_paths = {}
    
    for format in formats:
        filename = f"{room_name}_16states.{format}"
        filepath = os.path.join(data_dir, filename)
        
        if format == "npy":
            np.save(filepath, room_np)
        elif format == "pt":
            torch.save(room_tensor, filepath)
        elif format == "txt":
            np.savetxt(filepath, room_np, fmt='%d')
        else:
            print(f"Warning: Unsupported format '{format}', skipping")
            continue
            
        saved_paths[format] = filepath
    
    return saved_paths

def create_random_room(height: int, 
                      width: int, 
                      n_states: int = 16,
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Create a random room layout.
    
    Args:
        height: Room height
        width: Room width  
        n_states: Number of possible observation states
        seed: Random seed for reproducibility
        
    Returns:
        Random room layout as numpy array
        
    Examples:
        >>> room = create_random_room(20, 20, seed=42)
        >>> print(room.shape)  # (20, 20)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create random room with values 0 to n_states-1
    room = np.random.randint(0, n_states, size=(height, width), dtype=np.int64)
    
    return room

def validate_room_data(room_data: Union[np.ndarray, torch.Tensor]) -> bool:
    """
    Validate room data format and contents.
    
    Args:
        room_data: Room layout data to validate
        
    Returns:
        True if valid, raises ValueError if invalid
        
    Examples:
        >>> room = np.array([[0, 1], [2, 3]])
        >>> validate_room_data(room)  # True
    """
    # Convert to numpy for validation
    if isinstance(room_data, torch.Tensor):
        room_np = room_data.cpu().numpy()
    else:
        room_np = np.asarray(room_data)
    
    # Check basic properties
    if room_np.ndim != 2:
        raise ValueError(f"Room data must be 2D, got {room_np.ndim}D")
    
    if room_np.size == 0:
        raise ValueError("Room data cannot be empty")
    
    if not np.issubdtype(room_np.dtype, np.integer):
        raise ValueError(f"Room data must be integer type, got {room_np.dtype}")
    
    # Check value ranges
    min_val, max_val = room_np.min(), room_np.max()
    if min_val < -1:
        raise ValueError(f"Room values must be >= -1 (walls), got minimum {min_val}")
    
    if max_val > 15:
        raise ValueError(f"Room values must be <= 15, got maximum {max_val}")
    
    return True

def get_available_rooms(data_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Get list of available room datasets.
    
    Args:
        data_dir: Directory to search (defaults to package rooms/)
        
    Returns:
        Dictionary mapping room names to available file formats
        
    Examples:
        >>> rooms = get_available_rooms()
        >>> print(list(rooms.keys()))  # ['room_5x5', 'room_10x10', ...]
    """
    if data_dir is None:
        package_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(package_dir, "rooms")
    
    if not os.path.exists(data_dir):
        return {}
    
    rooms = {}
    
    # Scan for room files
    for filename in os.listdir(data_dir):
        if "_16states." in filename:
            # Extract room name and format
            base_name = filename.replace("_16states.", ".")
            room_name, ext = base_name.rsplit(".", 1)
            
            if room_name not in rooms:
                rooms[room_name] = {}
            
            rooms[room_name][ext] = os.path.join(data_dir, filename)
    
    return rooms

def room_info(room_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
    """
    Get information about room data.
    
    Args:
        room_data: Room layout data
        
    Returns:
        Dictionary with room statistics
        
    Examples:
        >>> info = room_info(room_array)
        >>> print(f"Room size: {info['shape']}")
    """
    # Convert to numpy for analysis
    if isinstance(room_data, torch.Tensor):
        room_np = room_data.cpu().numpy()
    else:
        room_np = np.asarray(room_data)
    
    # Count walls (value -1)
    n_walls = np.sum(room_np == -1)
    n_free = np.sum(room_np != -1)
    
    # Count unique observation types
    free_values = room_np[room_np != -1]
    unique_obs = len(np.unique(free_values)) if len(free_values) > 0 else 0
    
    info = {
        'shape': room_np.shape,
        'total_cells': room_np.size,
        'walls': int(n_walls),
        'free_cells': int(n_free),
        'wall_percentage': float(n_walls / room_np.size * 100),
        'unique_observations': int(unique_obs),
        'value_range': (int(room_np.min()), int(room_np.max())),
        'dtype': str(room_np.dtype)
    }
    
    return info