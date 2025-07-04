# Import base adapter first
from .base_adapter import CSCGEnvironmentAdapter, plot_graph

# Import room adapters with error handling
try:
    from .room_adapter import RoomAdapter
    # Backward compatibility
    RoomAdapter = RoomAdapter
    _ROOM_ADAPTERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import room adapters: {e}")
    RoomNPAdapter = None
    RoomTorchAdapter = None
    RoomAdapter = None
    save_room_plot = None
    _ROOM_ADAPTERS_AVAILABLE = False

# Import utilities with error handling
try:
    from .room_utils import (
        create_room_adapter, get_room_n_clones
    )
    _ROOM_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import room utilities: {e}")
    create_room_adapter = None
    get_room_n_clones = None
    demo_room_setup = None
    get_obs_colormap = None
    clone_to_obs_map = None
    top_k_used_clones = None
    count_used_clones = None
    _ROOM_UTILS_AVAILABLE = False

# Define what's available for import
__all__ = ["CSCGEnvironmentAdapter", "plot_graph"]

if _ROOM_ADAPTERS_AVAILABLE:
    __all__.extend(["RoomAdapter", "RoomNPAdapter", "RoomTorchAdapter", "save_room_plot"])

if _ROOM_UTILS_AVAILABLE:
    __all__.extend([
        "create_room_adapter", "get_room_n_clones", "demo_room_setup",
        "get_obs_colormap", "clone_to_obs_map", "top_k_used_clones", 
        "count_used_clones"
    ])