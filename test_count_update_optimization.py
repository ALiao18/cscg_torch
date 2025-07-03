#!/usr/bin/env python3
"""
Test script to validate and measure the count update optimization.
This should show significant speedup for the 37% bottleneck.
"""

import os
import sys
import time
import torch

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

def test_count_update_optimization():
    """Test the batched count update optimization."""
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Testing count update optimization on {device}")
    
    # Generate test data similar to L4 GPU test
    n_steps = 15000
    n_observations = 400
    clones_per_obs = 16
    n_actions = 4
    
    n_clones = torch.tensor([clones_per_obs] * n_observations, dtype=torch.int64, device=device)
    x = torch.randint(0, n_observations, (n_steps,), dtype=torch.int64, device=device)
    a = torch.randint(0, n_actions, (n_steps,), dtype=torch.int64, device=device)
    
    print(f"Test data: {n_steps:,} steps, {n_clones.sum().item():,} states")
    
    # Test original (sequential) implementation
    print("\n=== Testing Sequential Count Update ===")
    os.environ['CHMM_BATCHED_UPDATES'] = '0'  # Disable batching
    
    # Clear module cache to reload with new environment
    modules_to_clear = [mod for mod in sys.modules.keys() if 'models.' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    def test_sequential():
        from models.train_utils import train_chmm
        return train_chmm(n_clones, x, a, device=device, method='em_T', n_iter=1, seed=42)
    
    start = time.perf_counter()
    model_seq, prog_seq = test_sequential()
    time_sequential = time.perf_counter() - start
    
    print(f"Sequential count update: {time_sequential:.3f}s")
    
    # Test optimized (batched) implementation
    print("\n=== Testing Batched Count Update ===")
    os.environ['CHMM_BATCHED_UPDATES'] = '1'  # Enable batching
    
    # Clear module cache again
    modules_to_clear = [mod for mod in sys.modules.keys() if 'models.' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    def test_batched():
        from models.train_utils import train_chmm
        return train_chmm(n_clones, x, a, device=device, method='em_T', n_iter=1, seed=42)
    
    start = time.perf_counter()
    model_batch, prog_batch = test_batched()
    time_batched = time.perf_counter() - start
    
    print(f"Batched count update: {time_batched:.3f}s")
    
    # Calculate improvement
    speedup = time_sequential / time_batched
    improvement_pct = ((time_sequential - time_batched) / time_sequential) * 100
    
    print(f"\n🚀 COUNT UPDATE OPTIMIZATION RESULTS:")
    print(f"   Sequential: {time_sequential:.3f}s")
    print(f"   Batched: {time_batched:.3f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Improvement: {improvement_pct:.1f}%")
    
    # Verify results are identical
    bps_diff = abs(prog_seq[-1] - prog_batch[-1])
    print(f"\n✅ Results validation:")
    print(f"   Sequential BPS: {prog_seq[-1]:.6f}")
    print(f"   Batched BPS: {prog_batch[-1]:.6f}")
    print(f"   Difference: {bps_diff:.8f}")
    
    if bps_diff < 1e-6:
        print("   ✅ Results identical - optimization is correct")
    else:
        print("   ⚠️  Results differ - check optimization")
    
    # Project impact on 150k steps
    if speedup > 1.1:  # If we got meaningful speedup
        original_150k_time = 88.1  # minutes from L4 test
        # Count update was 37.4% of total time
        count_update_portion = original_150k_time * 0.374
        other_portions = original_150k_time * 0.626
        
        # Apply speedup to count update portion
        optimized_count_time = count_update_portion / speedup
        new_total_time = optimized_count_time + other_portions
        
        print(f"\n📊 Projected impact on 150k steps:")
        print(f"   Original total time: {original_150k_time:.1f} minutes")
        print(f"   Count update portion: {count_update_portion:.1f} minutes")
        print(f"   Optimized count time: {optimized_count_time:.1f} minutes")
        print(f"   New total time: {new_total_time:.1f} minutes")
        print(f"   Overall speedup: {original_150k_time/new_total_time:.2f}x")
        
        if new_total_time < 60:
            print(f"   🎯 Target achieved: Under 1 hour!")
        elif new_total_time < 120:
            print(f"   🎯 Excellent: Under 2 hours!")
        else:
            print(f"   📈 Good progress, more optimizations needed")
    
    return {
        'sequential_time': time_sequential,
        'batched_time': time_batched,
        'speedup': speedup,
        'improvement_pct': improvement_pct,
        'bps_diff': bps_diff
    }

def run_component_timing_analysis():
    """Analyze where time is spent in the optimized version."""
    print(f"\n=== Component Timing Analysis ===")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # Smaller test for component analysis
    n_steps = 5000
    n_observations = 100
    clones_per_obs = 16
    n_actions = 4
    
    n_clones = torch.tensor([clones_per_obs] * n_observations, dtype=torch.int64, device=device)
    x = torch.randint(0, n_observations, (n_steps,), dtype=torch.int64, device=device)
    a = torch.randint(0, n_actions, (n_steps,), dtype=torch.int64, device=device)
    
    os.environ['CHMM_BATCHED_UPDATES'] = '1'  # Use optimized version
    
    from models.chmm_torch import CHMM_torch
    from models.train_utils import forward, backward, updateC
    
    # Initialize model
    model = CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
    
    # Time each component
    timings = {}
    
    # Forward pass
    start = time.perf_counter()
    T_tr = model.T.transpose(1, 2)
    workspace = {}
    log2_lik, mess_fwd = forward(T_tr, model.Pi_x, model.n_clones, x, a, device, 
                                store_messages=True, workspace=workspace)
    timings['forward'] = time.perf_counter() - start
    
    # Backward pass
    start = time.perf_counter()
    mess_bwd = backward(model.T, model.n_clones, x, a, device, workspace=workspace)
    timings['backward'] = time.perf_counter() - start
    
    # Count update (the optimized part)
    start = time.perf_counter()
    C = torch.zeros_like(model.T)
    updateC(C, model.T, model.n_clones, mess_fwd, mess_bwd, x, a, device, workspace=workspace)
    timings['count_update'] = time.perf_counter() - start
    
    total_time = sum(timings.values())
    
    print(f"Component timing breakdown:")
    for component, time_val in timings.items():
        pct = (time_val / total_time) * 100
        print(f"  {component}: {time_val:.3f}s ({pct:.1f}%)")
    print(f"  Total: {total_time:.3f}s")
    
    return timings

if __name__ == "__main__":
    print("Count Update Optimization Test")
    print("=" * 50)
    
    try:
        results = test_count_update_optimization()
        
        if results['speedup'] > 1.5:
            print(f"\n🎉 Excellent optimization! {results['speedup']:.2f}x speedup achieved!")
        elif results['speedup'] > 1.1:
            print(f"\n✅ Good optimization! {results['speedup']:.2f}x speedup achieved!")
        else:
            print(f"\n📊 Minimal improvement. May need different approach.")
        
        # Run component analysis
        component_timings = run_component_timing_analysis()
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()