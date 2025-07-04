"""
Configuration file for CSCG-Torch Training Pipeline

Edit these parameters directly instead of using command line arguments.
"""

# Room and trajectory settings
ROOM_NAME = "room_5x5"  # Options: "room_5x5", "room_10x10", "room_20x20", "room_50x50"
TRAJECTORY_LENGTH = 15000  # Number of steps in the trajectory
SEED = 42  # Random seed for reproducibility

# Model parameters
N_CLONES_PER_OBS = 30  # Number of clones per observation (CPU is 20x faster for small problems)
PSEUDOCOUNT = 0.01  # Regularization pseudocount

# Training parameters
EM_ITERATIONS = 100  # Number of EM iterations
VITERBI_ITERATIONS = 50  # Number of Viterbi iterations

# Device settings  
DEVICE = "cpu"  # Options: "auto", "cuda", "mps", "cpu" - Using CPU for small problems like numba

# Debug and output settings
DEBUG_MODE = False  # Enable debug assertions
SAVE_RESULTS = True  # Save trained models and results
VERBOSE = True  # Print detailed progress

# Advanced settings
TERM_EARLY = True  # Terminate EM early if no improvement
ENABLE_BATCHED_UPDATES = True  # Use batched updateC optimization
ENABLE_PROFILING = False  # Enable performance profiling

# Results directory (will be created automatically)
RESULTS_BASE_DIR = "results"  # Base directory for saving results

# Experiment naming (auto-generated if None)
EXPERIMENT_NAME = None  # Will default to: {ROOM_NAME}_{TRAJECTORY_LENGTH}_{N_CLONES_PER_OBS}

# Visualization settings (for future use)
PLOT_CONVERGENCE = True  # Plot convergence curves
PLOT_ROOM = True  # Plot room layout with path
SAVE_PLOTS = True  # Save plots to file

# Performance optimization flags
SKIP_DEVICE_CHECKS = True  # Skip device checks for performance
USE_MIXED_PRECISION = False  # Use mixed precision training (experimental)

# Validation settings
VALIDATE_TRAJECTORY = True  # Validate generated trajectory
VALIDATE_MODEL = True  # Validate model after training

# Logging settings
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_TO_FILE = False  # Save logs to file
LOG_INTERVAL = 10  # Log every N iterations during training

def get_experiment_name():
    """Generate experiment name if not specified"""
    if EXPERIMENT_NAME is not None:
        return EXPERIMENT_NAME
    return f"{ROOM_NAME}_{TRAJECTORY_LENGTH}_{N_CLONES_PER_OBS}"

def validate_config():
    """Validate configuration parameters"""
    valid_rooms = ["room_5x5", "room_10x10", "room_20x20", "room_50x50"]
    assert ROOM_NAME in valid_rooms, f"ROOM_NAME must be one of {valid_rooms}"
    
    assert TRAJECTORY_LENGTH > 0, "TRAJECTORY_LENGTH must be positive"
    assert N_CLONES_PER_OBS > 0, "N_CLONES_PER_OBS must be positive"
    assert PSEUDOCOUNT >= 0, "PSEUDOCOUNT must be non-negative"
    assert EM_ITERATIONS > 0, "EM_ITERATIONS must be positive"
    assert VITERBI_ITERATIONS > 0, "VITERBI_ITERATIONS must be positive"
    
    valid_devices = ["auto", "cuda", "mps", "cpu"]
    assert DEVICE in valid_devices, f"DEVICE must be one of {valid_devices}"
    
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    assert LOG_LEVEL in valid_log_levels, f"LOG_LEVEL must be one of {valid_log_levels}"
    
    print("✓ Configuration validation passed")

def print_config():
    """Print current configuration"""
    print("Current Configuration:")
    print("=" * 50)
    print(f"Room: {ROOM_NAME}")
    print(f"Trajectory Length: {TRAJECTORY_LENGTH}")
    print(f"Clones per Observation: {N_CLONES_PER_OBS}")
    print(f"EM Iterations: {EM_ITERATIONS}")
    print(f"Viterbi Iterations: {VITERBI_ITERATIONS}")
    print(f"Device: {DEVICE}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print(f"Save Results: {SAVE_RESULTS}")
    print(f"Experiment Name: {get_experiment_name()}")
    print("=" * 50)

if __name__ == "__main__":
    validate_config()
    print_config()