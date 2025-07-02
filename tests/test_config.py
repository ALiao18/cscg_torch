"""
Test Configuration and Utilities

Shared configuration and utility functions for CSCG testing.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

# Test configuration
class TestConfig:
    """Configuration settings for CSCG tests."""
    
    # Default test parameters
    SEQUENCE_LENGTH = 2000
    ROOM_SIZE = 5  # Use 5x5 for fast iteration
    N_CLONES_PER_OBS = 2
    EM_ITERATIONS = 10
    SEED = 42
    
    # Device settings
    USE_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    
    # Numerical tolerances
    FLOAT_TOL = 1e-6
    CONVERGENCE_TOL = 1e-4
    
    # Test room specifications
    ROOM_SIZES = [5, 10, 20, 50]
    N_STATES = 16
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    ROOMS_DIR = PROJECT_ROOT / "rooms"
    TEST_DATA_DIR = Path(__file__).parent / "data"

def setup_test_environment():
    """
    Set up the test environment with consistent settings.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(TestConfig.SEED)
    np.random.seed(TestConfig.SEED)
    
    if TestConfig.USE_GPU:
        torch.cuda.manual_seed(TestConfig.SEED)
        torch.cuda.manual_seed_all(TestConfig.SEED)
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Suppress warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning)

def load_test_room(size: int = None) -> torch.Tensor:
    """
    Load a test room layout.
    
    Args:
        size: Room size (default: TestConfig.ROOM_SIZE)
        
    Returns:
        torch.Tensor: Room layout tensor
    """
    if size is None:
        size = TestConfig.ROOM_SIZE
    
    room_path = TestConfig.ROOMS_DIR / f"room_{size}x{size}_{TestConfig.N_STATES}states.pt"
    
    if not room_path.exists():
        raise FileNotFoundError(f"Test room not found: {room_path}")
    
    return torch.load(room_path)

def create_test_sequence(length: int = None, n_obs: int = None, 
                        n_actions: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a test observation and action sequence.
    
    Args:
        length: Sequence length (default: TestConfig.SEQUENCE_LENGTH)
        n_obs: Number of observation types (default: TestConfig.N_STATES)
        n_actions: Number of action types (default: 4)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (observations, actions)
    """
    if length is None:
        length = TestConfig.SEQUENCE_LENGTH
    if n_obs is None:
        n_obs = TestConfig.N_STATES
    
    # Generate random but valid sequences
    x = torch.randint(0, n_obs, (length,), dtype=torch.int64)
    a = torch.randint(0, n_actions, (length,), dtype=torch.int64)
    
    return x, a

def create_test_n_clones(n_obs: int = None, n_clones_per_obs: int = None) -> torch.Tensor:
    """
    Create n_clones tensor for testing.
    
    Args:
        n_obs: Number of observation types (default: TestConfig.N_STATES)
        n_clones_per_obs: Clones per observation (default: TestConfig.N_CLONES_PER_OBS)
        
    Returns:
        torch.Tensor: n_clones tensor
    """
    if n_obs is None:
        n_obs = TestConfig.N_STATES
    if n_clones_per_obs is None:
        n_clones_per_obs = TestConfig.N_CLONES_PER_OBS
    
    return torch.full((n_obs,), n_clones_per_obs, dtype=torch.int64)

def check_tensor_properties(tensor: torch.Tensor, expected_shape: Optional[Tuple] = None,
                          expected_dtype: Optional[torch.dtype] = None,
                          expected_device: Optional[torch.device] = None,
                          finite_check: bool = True) -> bool:
    """
    Check tensor properties for validation.
    
    Args:
        tensor: Tensor to check
        expected_shape: Expected tensor shape
        expected_dtype: Expected tensor dtype
        expected_device: Expected tensor device
        finite_check: Whether to check for finite values
        
    Returns:
        bool: True if all checks pass
    """
    try:
        assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
        
        if expected_shape is not None:
            assert tensor.shape == expected_shape, f"Shape mismatch: {tensor.shape} != {expected_shape}"
        
        if expected_dtype is not None:
            assert tensor.dtype == expected_dtype, f"Dtype mismatch: {tensor.dtype} != {expected_dtype}"
        
        if expected_device is not None:
            assert tensor.device == expected_device, f"Device mismatch: {tensor.device} != {expected_device}"
        
        if finite_check:
            assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"
        
        return True
    except AssertionError as e:
        print(f"Tensor property check failed: {e}")
        return False

def measure_performance(func, *args, **kwargs) -> Tuple[any, float]:
    """
    Measure function execution time.
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple[any, float]: (function_result, execution_time_seconds)
    """
    import time
    
    # GPU synchronization if needed
    if TestConfig.USE_GPU:
        torch.cuda.synchronize()
    
    start_time = time.time()
    result = func(*args, **kwargs)
    
    if TestConfig.USE_GPU:
        torch.cuda.synchronize()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time

def check_gpu_memory() -> dict:
    """
    Check GPU memory usage if CUDA is available.
    
    Returns:
        dict: Memory usage information
    """
    if not TestConfig.USE_GPU:
        return {"gpu_available": False}
    
    return {
        "gpu_available": True,
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }

def cleanup_test_environment():
    """
    Clean up test environment after tests.
    """
    if TestConfig.USE_GPU:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# Test decorators
def gpu_test(func):
    """Decorator to mark tests that require GPU."""
    def wrapper(*args, **kwargs):
        if not TestConfig.USE_GPU:
            print(f"Skipping GPU test {func.__name__} (CUDA not available)")
            return
        return func(*args, **kwargs)
    return wrapper

def slow_test(func):
    """Decorator to mark slow tests."""
    def wrapper(*args, **kwargs):
        print(f"Running slow test {func.__name__}...")
        return func(*args, **kwargs)
    return wrapper