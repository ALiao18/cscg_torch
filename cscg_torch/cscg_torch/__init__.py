"""
CSCG-Torch: Compositional State-Action Graph Models in PyTorch

A PyTorch implementation of Compositional State-Action Graph (CSCG) models 
for reinforcement learning and navigation tasks, optimized for modern GPUs.

Key Features:
- GPU-accelerated sequence generation with CUDA/MPS support
- V100/A100 optimized Tensor Core operations  
- Mixed-precision training for improved performance
- Modular design for easy integration and extension

Quick Start:
    >>> import cscg_torch
    >>> from cscg_torch import create_room_adapter, train_chmm
    >>> from cscg_torch.utils import load_room_data
    
    # Load or create room data
    >>> room_data = load_room_data("room_20x20")
    
    # Create adapter and generate sequences
    >>> adapter = create_room_adapter(room_data)
    >>> x_seq, a_seq = adapter.generate_sequence_gpu(10000)
    
    # Train CHMM model
    >>> model, progression = train_chmm(n_clones, x_seq, a_seq)

GPU Optimization:
    The package automatically detects and optimizes for:
    - V100 GPUs: 32K-64K chunk sizes, FP16 Tensor Cores
    - A100/H100 GPUs: 64K-128K chunk sizes, advanced mixed precision
    - MPS (Apple Silicon): 8K chunk sizes, unified memory optimization
    - CPU fallback with optimized numpy operations

Google Colab Usage:
    !pip install cscg-torch
    import cscg_torch
    # Ready to use!
"""

__version__ = "1.0.0"
__author__ = "Andrew Liao"
__email__ = "yl8520@nyu.edu"

# Core imports for easy access
from .models.chmm_torch import CHMM_torch
from .models.train_utils import (
    train_chmm, make_E, make_E_sparse, compute_forward_messages, place_field
)
from .env_adapters.room_utils import (
    create_room_adapter, get_room_n_clones, generate_room_sequence
)
from .env_adapters.room_adapter import RoomTorchAdapter, RoomNPAdapter
from .env_adapters.base_adapter import CSCGEnvironmentAdapter, plot_graph

# Utility imports
from .utils import (
    load_room_data, 
    save_room_data,
    plot_training_progression,
    plot_room_layout,
    detect_optimal_device,
    get_gpu_info,
    optimize_for_gpu,
    benchmark_device,
    get_memory_info
)
from .utils.data_utils import create_random_room, room_info, get_available_rooms

# Make key classes and functions available at package level
__all__ = [
    # Core models (maintain backward compatibility)
    'CHMM_torch',
    'train_chmm',
    'make_E',
    'make_E_sparse', 
    'compute_forward_messages',
    'place_field',
    
    # Environment adapters
    'create_room_adapter',
    'RoomTorchAdapter', 
    'RoomNPAdapter',
    'CSCGEnvironmentAdapter',
    'generate_room_sequence',
    'get_room_n_clones',
    
    # Utilities
    'load_room_data',
    'save_room_data',
    'create_random_room',
    'room_info',
    'get_available_rooms',
    'plot_training_progression', 
    'plot_room_layout',
    'plot_graph',
    'detect_optimal_device',
    'get_gpu_info',
    'optimize_for_gpu',
    'benchmark_device',
    'get_memory_info',
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]

# GPU optimization setup
import os
import torch

# Set optimal defaults for different hardware
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
elif torch.backends.mps.is_available():
    # MPS optimizations
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

# Optional verbose import (uncomment for debugging)
def _print_system_info():
    """Print system information on package import."""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f" CSCG-Torch v{__version__} ready! GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        elif torch.backends.mps.is_available():
            print(f" CSCG-Torch v{__version__} ready! Using Apple Silicon MPS")
        else:
            print(f" CSCG-Torch v{__version__} ready! Using CPU")
    except:
        print(f" CSCG-Torch v{__version__} loaded")

# Uncomment to enable verbose import messages
# _print_system_info()