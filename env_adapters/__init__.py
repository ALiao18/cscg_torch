
from .base_adapter import CSCGEnvironmentAdapter
from .room_adapter import RoomNPAdapter, RoomTorchAdapter
from .room_utils import create_room_adapter, get_room_n_clones, demo_room_setup

# Backward compatibility
RoomAdapter = RoomTorchAdapter

__all__ = [
    "CSCGEnvironmentAdapter",
    "RoomAdapter",
    "RoomNPAdapter", 
    "RoomTorchAdapter",
    "create_room_adapter",
    "get_room_n_clones",
    "demo_room_setup"
]