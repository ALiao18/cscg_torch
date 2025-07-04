"""
A100 GPU Setup Diagnostic Script

This script verifies:
1. CUDA availability and device properties
2. CUDA kernel compilation
3. Basic CHMM operations
4. Memory management
5. Performance benchmarking
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.chmm_optimized import CHMM_Optimized
from models.cuda_kernels import get_cuda_kernels

def print_device_info():
    """Print detailed information about available CUDA devices"""
    print("\n=== CUDA Device Information ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
        
    device_count = torch.cuda.device_count()
    print(f"✓ Found {device_count} CUDA device(s)")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - CUDA Arch: sm_{props.major}{props.minor}")
        print(f"  - Multiprocessors: {props.multi_processor_count}")
        
        # Check if it's an A100
        if "A100" in props.name:
            print("\n✓ A100 GPU detected!")
            print("  - Architecture: Ampere")
            print("  - Tensor Cores: 3rd Generation")
            print("  - NVLink: 3rd Generation")
            print("  - Multi-Instance GPU (MIG): Supported")
            print(f"  - SMs: {props.multi_processor_count}")
    
    # Print CUDA version information
    print(f"\nCUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if cuDNN is enabled
    print(f"cuDNN Enabled: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    # Print current device settings
    print("\nCurrent Device Settings:")
    print(f"  - Current Device: {torch.cuda.current_device()}")
    print(f"  - Device Count: {torch.cuda.device_count()}")
    print(f"  - Default Tensor Type: {torch.get_default_dtype()}")
    
    return True

def verify_cuda_compilation():
    """Verify CUDA kernel compilation"""
    print("\n=== CUDA Kernel Compilation ===")
    
    try:
        cuda_kernels = get_cuda_kernels()
        print("✓ CUDA kernels compiled successfully")
        return True
    except Exception as e:
        print(f"❌ CUDA kernel compilation failed: {e}")
        print("\nTrying to install build tools...")
        try:
            import subprocess
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "ninja-build"], check=True)
            print("✓ Build tools installed")
            
            # Try compilation again
            cuda_kernels = get_cuda_kernels()
            print("✓ CUDA kernels compiled successfully after installing build tools")
            return True
        except Exception as install_error:
            print(f"❌ Failed to install build tools: {install_error}")
            return False

def test_basic_operations(seq_len=1000, n_states=100):
    """Test basic CHMM operations"""
    print("\n=== Basic CHMM Operations ===")
    
    device = torch.device('cuda')
    
    # Generate test data
    n_obs = 10
    n_actions = 4
    n_clones = torch.ones(n_obs, dtype=torch.int64, device=device) * (n_states // n_obs)
    x = torch.randint(0, n_obs, (seq_len,), device=device)
    a = torch.randint(0, n_actions, (seq_len,), device=device)
    
    try:
        # Create model
        model = CHMM_Optimized(
            n_clones=n_clones,
            x=x, 
            a=a,
            pseudocount=0.01,
            device=device,
            enable_mixed_precision=True  # Enable for A100
        )
        print("✓ Model created successfully")
        
        # Test forward pass
        start_time = time.time()
        bps = model.bps_optimized(x, a)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        print(f"✓ Forward pass: {forward_time*1000:.1f}ms")
        
        # Test EM iteration
        start_time = time.time()
        convergence = model.learn_em_T_optimized(x, a, n_iter=1)
        torch.cuda.synchronize()
        em_time = time.time() - start_time
        print(f"✓ EM iteration: {em_time*1000:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Operation test failed: {e}")
        return False

def check_memory_usage():
    """Monitor GPU memory usage"""
    print("\n=== GPU Memory Usage ===")
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Current allocated: {allocated:.2f} GB")
    print(f"Current reserved: {reserved:.2f} GB")
    print(f"Peak allocated: {max_allocated:.2f} GB")

def main():
    print("🔍 Running A100 Setup Diagnostics")
    print("="*50)
    
    success = True
    
    # Step 1: Check CUDA device
    if not print_device_info():
        print("\n❌ No CUDA device available - cannot proceed")
        return
    
    # Step 2: Verify CUDA compilation
    if not verify_cuda_compilation():
        print("\n⚠️  CUDA kernel compilation failed - falling back to PyTorch")
        success = False
    
    # Step 3: Test operations
    if not test_basic_operations():
        print("\n⚠️  Basic operations failed")
        success = False
    
    # Step 4: Check memory
    check_memory_usage()
    
    print("\n=== Final Status ===")
    if success:
        print("✅ All checks passed - A100 setup is ready!")
    else:
        print("⚠️  Some checks failed - see above for details")

if __name__ == "__main__":
    main() 