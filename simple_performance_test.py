#!/usr/bin/env python3
"""
Simple performance test to identify bottlenecks in CHMM training.
Generate 150k steps and profile 1 training iteration.
"""

import os
import sys
import time
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

# Disable debug mode for performance testing
os.environ['CHMM_DEBUG'] = '0'

# Set up for GPU optimization
torch.backends.cudnn.benchmark = True

def detect_device():
    """Detect optimal device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def generate_test_data(n_steps=150000, device=None):
    """Generate synthetic test data."""
    print(f"Generating {n_steps:,} steps of synthetic data...")
    
    if device is None:
        device = detect_device()
    
    # Create synthetic room data (20x20 grid)
    room_size = 20
    n_observations = room_size * room_size  # 400 observations
    n_actions = 4  # up, down, left, right
    
    # Each observation has 16 clones (typical for room navigation)
    n_clones = torch.tensor([16] * n_observations, dtype=torch.int64, device=device)
    
    # Generate random sequences
    torch.manual_seed(42)
    x = torch.randint(0, n_observations, (n_steps,), dtype=torch.int64, device=device)
    a = torch.randint(0, n_actions, (n_steps,), dtype=torch.int64, device=device)
    
    print(f"Generated data:")
    print(f"  Sequence length: {len(x):,}")
    print(f"  Total states: {n_clones.sum().item():,}")
    print(f"  Actions: {n_actions}")
    print(f"  Device: {device}")
    
    return n_clones, x, a

def time_operation(name, func, *args, **kwargs):
    """Time an operation with proper GPU synchronization."""
    device = kwargs.get('device', torch.device('cpu'))
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    result = func(*args, **kwargs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s")
    
    return result, elapsed

def test_assert_overhead(device):
    """Test assert statement overhead."""
    print("\n=== Assert Statement Overhead Test ===")
    
    # Small test tensors
    n_states = 1000
    seq_len = 1000
    
    x = torch.randint(0, 25, (seq_len,), dtype=torch.int64, device=device)
    a = torch.randint(0, 4, (seq_len,), dtype=torch.int64, device=device)
    n_clones = torch.tensor([40] * 25, dtype=torch.int64, device=device)
    
    # Enable debug mode
    os.environ['CHMM_DEBUG'] = '1'
    
    def validate_with_asserts():
        from models.train_utils import validate_seq
        validate_seq(x, a, n_clones)
        return True
    
    _, time_with_asserts = time_operation("validate_with_asserts", validate_with_asserts)
    
    # Disable debug mode
    os.environ['CHMM_DEBUG'] = '0'
    
    def validate_without_asserts():
        from models.train_utils import validate_seq
        validate_seq(x, a, n_clones)
        return True
    
    _, time_without_asserts = time_operation("validate_without_asserts", validate_without_asserts)
    
    overhead = time_with_asserts - time_without_asserts
    if time_without_asserts > 0:
        overhead_pct = (overhead / time_without_asserts) * 100
        print(f"Assert overhead: {overhead:.4f}s ({overhead_pct:.1f}%)")
    
    return overhead

def test_device_transfers(n_clones, x, a, device):
    """Check for device transfer issues."""
    print("\n=== Device Transfer Test ===")
    
    # Check if all tensors are on the same device
    devices = [
        ("n_clones", n_clones.device),
        ("x", x.device),
        ("a", a.device),
        ("target", device)
    ]
    
    print("Tensor devices:")
    all_same = True
    for name, tensor_device in devices:
        print(f"  {name}: {tensor_device}")
        if tensor_device != device:
            all_same = False
    
    if all_same:
        print("✅ All tensors on same device")
    else:
        print("❌ Device mismatch detected")
        
    # Test transfer cost
    if device.type == 'cuda':
        # Move to CPU and back
        def transfer_test():
            x_cpu = x.cpu()
            x_gpu = x_cpu.to(device)
            return x_gpu
        
        _, transfer_time = time_operation("cpu_gpu_transfer", transfer_test)
        transfer_gb = (x.numel() * x.element_size()) / (1024**3)
        print(f"Transfer speed: {transfer_gb/transfer_time:.1f} GB/s")
    
    return all_same

def profile_training_components(n_clones, x, a, device):
    """Profile individual training components."""
    print("\n=== Training Component Profiling ===")
    
    from models.chmm_torch import CHMM_torch
    from models.train_utils import forward, backward, updateC
    
    # Initialize model
    def init_model():
        return CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
    
    model, init_time = time_operation("model_init", init_model)
    
    # Forward pass
    def forward_pass():
        T_tr = model.T.transpose(1, 2)
        return forward(T_tr, model.Pi_x, model.n_clones, x, a, device, 
                      store_messages=True, workspace={})
    
    (log2_lik, mess_fwd), fwd_time = time_operation("forward_pass", forward_pass)
    
    # Backward pass
    def backward_pass():
        return backward(model.T, model.n_clones, x, a, device, workspace={})
    
    mess_bwd, bwd_time = time_operation("backward_pass", backward_pass)
    
    # Update counts
    def update_counts():
        C = torch.zeros_like(model.T)
        updateC(C, model.T, model.n_clones, mess_fwd, mess_bwd, x, a, device, workspace={})
        return C
    
    C, update_time = time_operation("update_counts", update_counts)
    
    # Full training iteration
    def full_iteration():
        from models.train_utils import train_chmm
        return train_chmm(n_clones, x, a, device=device, method='em_T', n_iter=1, 
                         pseudocount=0.01, seed=42)
    
    (trained_model, progression), full_time = time_operation("full_iteration", full_iteration)
    
    # Analysis
    print(f"\nComponent breakdown:")
    print(f"  Model init: {init_time:.3f}s ({init_time/full_time*100:.1f}%)")
    print(f"  Forward:    {fwd_time:.3f}s ({fwd_time/full_time*100:.1f}%)")
    print(f"  Backward:   {bwd_time:.3f}s ({bwd_time/full_time*100:.1f}%)")
    print(f"  Update:     {update_time:.3f}s ({update_time/full_time*100:.1f}%)")
    print(f"  Total:      {full_time:.3f}s")
    
    # Theoretical analysis
    seq_len = len(x)
    n_states = n_clones.sum().item()
    
    print(f"\nTheoretical analysis:")
    print(f"  Sequence length: {seq_len:,}")
    print(f"  States: {n_states:,}")
    print(f"  Ops per step: ~{int(np.sqrt(n_states)):,}")
    print(f"  Total ops: ~{seq_len * int(np.sqrt(n_states))**2:,}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  GPU: {gpu_name}")
        print(f"  Peak memory: {memory_gb:.1f}GB")
    
    return {
        'init_time': init_time,
        'forward_time': fwd_time,
        'backward_time': bwd_time,
        'update_time': update_time,
        'total_time': full_time,
        'seq_len': seq_len,
        'n_states': n_states
    }

def main():
    """Main performance testing function."""
    print("CHMM Performance Analysis")
    print("=" * 40)
    
    # Detect device
    device = detect_device()
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    
    # Generate test data
    n_clones, x, a = generate_test_data(150000, device)
    
    # Run tests
    assert_overhead = test_assert_overhead(device)
    device_ok = test_device_transfers(n_clones, x, a, device)
    results = profile_training_components(n_clones, x, a, device)
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"Total training time: {results['total_time']:.1f}s")
    print(f"Assert overhead: {assert_overhead:.4f}s")
    print(f"Device transfers: {'✅ OK' if device_ok else '❌ Issues'}")
    
    # Identify bottlenecks
    times = [
        ('Forward', results['forward_time']),
        ('Backward', results['backward_time']),
        ('Update', results['update_time']),
        ('Init', results['init_time'])
    ]
    times.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nBottlenecks (largest first):")
    for name, time_val in times:
        pct = (time_val / results['total_time']) * 100
        print(f"  {name}: {time_val:.3f}s ({pct:.1f}%)")
    
    # Recommendations
    print(f"\nRecommendations:")
    if results['total_time'] > 180:
        print("❌ Training time > 3 minutes - significant optimization needed")
    elif results['total_time'] > 60:
        print("⚠️  Training time > 1 minute - some optimization beneficial")
    else:
        print("✅ Training time acceptable")
    
    if assert_overhead > 0.1:
        print(f"⚠️  Assert overhead significant ({assert_overhead:.3f}s)")
    
    if not device_ok:
        print("❌ Fix device transfer issues")
    
    # A100 expectations
    if device.type == 'cuda':
        print(f"\nA100 Performance Analysis:")
        # A100 has ~19.5 TFLOPS FP32, ~1.6TB/s memory bandwidth
        theoretical_min = results['seq_len'] * results['n_states'] / (19.5e12)  # very rough estimate
        print(f"  Theoretical minimum: ~{theoretical_min:.1f}s")
        print(f"  Actual: {results['total_time']:.1f}s") 
        print(f"  Efficiency: ~{theoretical_min/results['total_time']*100:.1f}%")
    
    return results

if __name__ == "__main__":
    results = main()