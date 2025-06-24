"""
Google Colab Setup Utilities

Provides helper functions for setting up CSCG PyTorch in Google Colab environment.
"""

import torch
import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required packages in Colab environment."""
    packages = [
        "torch>=2.1.0",
        "numpy>=1.24", 
        "matplotlib>=3.7",
        "tqdm>=4.65",
        "scipy>=1.10"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def check_gpu():
    """Check GPU availability and print device information."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {device_name}")
        print(f"✓ GPU Memory: {memory_gb:.1f} GB")
        return True
    else:
        print("✗ No GPU available - using CPU")
        return False

def setup_colab_environment():
    """Complete setup for Google Colab."""
    print("Setting up CSCG PyTorch for Google Colab...")
    
    # Install dependencies
    print("\n1. Installing dependencies...")
    install_dependencies()
    
    # Check GPU
    print("\n2. Checking GPU availability...")
    gpu_available = check_gpu()
    
    # Set environment variables
    if gpu_available:
        os.environ['CSCG_DEVICE'] = 'cuda'
    else:
        os.environ['CSCG_DEVICE'] = 'cpu'
    
    # Add current directory to path
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    print("\n3. Setting up imports...")
    try:
        from cscg_torch import CHMM_torch
        print("✓ CSCG PyTorch successfully imported")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def optimize_for_colab():
    """Apply Colab-specific optimizations."""
    # Set environment variables for better performance
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    # Configure PyTorch for Colab
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    print("✓ Applied Colab optimizations")

def create_colab_example():
    """Create a simple example for testing in Colab."""
    example_code = '''
# CSCG PyTorch Example for Google Colab
import torch
import numpy as np
from cscg_torch import CHMM_torch

# Generate sample data
np.random.seed(42)
n_clones = torch.tensor([2, 3, 2], dtype=torch.int64)
x = torch.randint(0, 3, (100,), dtype=torch.int64)
a = torch.randint(0, 2, (99,), dtype=torch.int64)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CHMM_torch(n_clones, x, a, dtype=torch.float32)

# Quick training
print("Training model...")
convergence = model.learn_em_T(x, a, n_iter=10)
print(f"Final BPS: {convergence[-1]:.3f}")

# Test inference
bps = model.bps(x, a)
print(f"Test BPS: {bps:.3f}")

print("✓ Example completed successfully!")
'''
    
    with open('colab_example.py', 'w') as f:
        f.write(example_code)
    
    print("✓ Created example file: colab_example.py")

if __name__ == "__main__":
    # Run full setup
    success = setup_colab_environment()
    if success:
        optimize_for_colab()
        create_colab_example()
        print("\n🎉 Colab setup complete! Run 'exec(open('colab_example.py').read())' to test.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")