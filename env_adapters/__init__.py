
# Import base adapter first
from .base_adapter import CSCGEnvironmentAdapter

# Import room adapters with error handling
try:
    from .room_adapter import RoomNPAdapter, RoomTorchAdapter
    # Backward compatibility
    RoomAdapter = RoomTorchAdapter
    _ROOM_ADAPTERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import room adapters: {e}")
    RoomNPAdapter = None
    RoomTorchAdapter = None
    RoomAdapter = None
    _ROOM_ADAPTERS_AVAILABLE = False

# Import utilities with error handling
try:
    from .room_utils import create_room_adapter, get_room_n_clones, demo_room_setup
    _ROOM_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import room utilities: {e}")
    create_room_adapter = None
    get_room_n_clones = None
    demo_room_setup = None
    _ROOM_UTILS_AVAILABLE = False

# Define what's available for import
__all__ = ["CSCGEnvironmentAdapter"]

if _ROOM_ADAPTERS_AVAILABLE:
    __all__.extend(["RoomAdapter", "RoomNPAdapter", "RoomTorchAdapter"])

if _ROOM_UTILS_AVAILABLE:
    __all__.extend(["create_room_adapter", "get_room_n_clones", "demo_room_setup"])