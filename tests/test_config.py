import torch
import numpy as np
import warnings

def setup_test_environment(seed: int = 42):
    """
    Set up the test environment with consistent settings.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        torch.mps.empty_cache()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    warnings.filterwarnings("ignore", category=UserWarning)

def cleanup_test_environment():
    """
    Clean up test environment after tests.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats() # Removed as it causes issues on MPS
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def create_test_sequence(length: int = 100, n_obs: int = 16, n_actions: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a test observation and action sequence.
    """
    x = torch.randint(0, n_obs, (length,), dtype=torch.int64)
    a = torch.randint(0, n_actions, (length,), dtype=torch.int64)
    
    return x, a

def create_test_n_clones(n_obs: int = 16, n_clones_per_obs: int = 2) -> torch.Tensor:
    """
    Create n_clones tensor for testing.
    """
    return torch.full((n_obs,), n_clones_per_obs, dtype=torch.int64)
