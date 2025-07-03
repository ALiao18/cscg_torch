#!/usr/bin/env python3
"""
Test mixed precision (FP16) optimization for L4 GPU.
L4 has excellent Tensor Core support for FP16 which should provide 2x speedup.
"""

import os
import sys
import time
import torch

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

def test_mixed_precision_optimization():
    """Test mixed precision (FP16) optimization."""
    
    if not torch.cuda.is_available():
        print("CUDA not available - mixed precision requires CUDA")
        return
        
    device = torch.device("cuda")
    print(f"Testing mixed precision optimization on {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Check if FP16 is supported
    if torch.cuda.get_device_capability()[0] < 7:
        print("Warning: GPU may not have optimal FP16 support")
    else:
        print("✅ GPU has Tensor Core support for FP16")
    
    # Generate test data
    n_steps = 15000
    n_observations = 400
    clones_per_obs = 16
    n_actions = 4
    
    n_clones = torch.tensor([clones_per_obs] * n_observations, dtype=torch.int64, device=device)
    x = torch.randint(0, n_observations, (n_steps,), dtype=torch.int64, device=device)
    a = torch.randint(0, n_actions, (n_steps,), dtype=torch.int64, device=device)
    
    print(f"Test data: {n_steps:,} steps, {n_clones.sum().item():,} states")
    
    # Test FP32 (original precision)
    print("\n=== Testing FP32 (Original) ===")
    os.environ['CHMM_MIXED_PRECISION'] = '0'  # Disable mixed precision
    
    # Clear module cache
    modules_to_clear = [mod for mod in sys.modules.keys() if 'models.' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    def test_fp32():
        from models.train_utils import train_chmm
        return train_chmm(n_clones, x, a, device=device, method='em_T', n_iter=1, seed=42)
    
    # Warm up GPU
    _ = test_fp32()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    model_fp32, prog_fp32 = test_fp32()
    torch.cuda.synchronize()
    time_fp32 = time.perf_counter() - start
    
    fp32_memory = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()
    
    print(f"FP32 time: {time_fp32:.3f}s")
    print(f"FP32 memory: {fp32_memory:.2f}GB")
    
    # Test FP16 (mixed precision)
    print("\n=== Testing FP16 (Mixed Precision) ===")
    os.environ['CHMM_MIXED_PRECISION'] = '1'  # Enable mixed precision
    
    # Clear module cache
    modules_to_clear = [mod for mod in sys.modules.keys() if 'models.' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    def test_fp16():
        from models.train_utils import train_chmm
        return train_chmm(n_clones, x, a, device=device, method='em_T', n_iter=1, seed=42)
    
    # Warm up GPU
    _ = test_fp16()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    model_fp16, prog_fp16 = test_fp16()
    torch.cuda.synchronize()
    time_fp16 = time.perf_counter() - start
    
    fp16_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"FP16 time: {time_fp16:.3f}s")
    print(f"FP16 memory: {fp16_memory:.2f}GB")
    
    # Calculate improvements
    speedup = time_fp32 / time_fp16
    memory_reduction = (fp32_memory - fp16_memory) / fp32_memory * 100
    improvement_pct = ((time_fp32 - time_fp16) / time_fp32) * 100
    
    print(f"\n🚀 MIXED PRECISION OPTIMIZATION RESULTS:")
    print(f"   FP32: {time_fp32:.3f}s, {fp32_memory:.2f}GB")
    print(f"   FP16: {time_fp16:.3f}s, {fp16_memory:.2f}GB")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Memory reduction: {memory_reduction:.1f}%")
    print(f"   Time improvement: {improvement_pct:.1f}%")
    
    # Verify numerical accuracy
    bps_diff = abs(prog_fp32[-1] - prog_fp16[-1])
    relative_error = bps_diff / abs(prog_fp32[-1]) * 100
    
    print(f"\n🔍 Numerical Accuracy Check:")
    print(f"   FP32 BPS: {prog_fp32[-1]:.6f}")
    print(f"   FP16 BPS: {prog_fp16[-1]:.6f}")
    print(f"   Absolute diff: {bps_diff:.8f}")
    print(f"   Relative error: {relative_error:.4f}%")
    
    if relative_error < 0.1:
        print("   ✅ Excellent numerical accuracy")
    elif relative_error < 1.0:
        print("   ✅ Good numerical accuracy")
    elif relative_error < 5.0:
        print("   ⚠️  Acceptable accuracy for this application")
    else:
        print("   ❌ Accuracy may be too low")
    
    # Project impact on 150k steps
    if speedup > 1.1:
        original_150k_time = 88.1  # minutes from L4 test
        new_total_time = original_150k_time / speedup
        
        print(f"\n📊 Projected impact on 150k steps:")
        print(f"   Original time: {original_150k_time:.1f} minutes")
        print(f"   With FP16: {new_total_time:.1f} minutes")
        print(f"   Time saved: {original_150k_time - new_total_time:.1f} minutes")
        
        if new_total_time < 45:
            print(f"   🎯 Excellent! Under 45 minutes!")
        elif new_total_time < 60:
            print(f"   🎯 Great! Under 1 hour!")
        elif new_total_time < 120:
            print(f"   ✅ Good! Under 2 hours!")
        else:
            print(f"   📈 Progress made, more optimizations available")
    
    return {
        'fp32_time': time_fp32,
        'fp16_time': time_fp16,
        'speedup': speedup,
        'memory_reduction': memory_reduction,
        'relative_error': relative_error
    }

if __name__ == "__main__":
    print("Mixed Precision (FP16) Optimization Test")
    print("=" * 50)
    
    try:
        results = test_mixed_precision_optimization()
        
        if results and results['speedup'] > 1.5:
            print(f"\n🎉 Excellent optimization! {results['speedup']:.2f}x speedup with FP16!")
            print(f"Mixed precision is highly effective on this GPU")
        elif results and results['speedup'] > 1.2:
            print(f"\n✅ Good optimization! {results['speedup']:.2f}x speedup with FP16!")
        else:
            print(f"\n📊 Mixed precision may not be optimal for this workload")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()