#!/usr/bin/env python3
"""
Test the device check optimization impact on performance.
Compare training with and without device checks.
"""

import os
import sys
import time
import torch

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

def test_device_optimization():
    """Test performance impact of skipping device checks."""
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Testing device optimization on {device}")
    
    # Create test data (smaller for quick test)
    n_steps = 10000
    n_clones = torch.tensor([16] * 25, dtype=torch.int64, device=device)
    x = torch.randint(0, 25, (n_steps,), dtype=torch.int64, device=device)
    a = torch.randint(0, 4, (n_steps,), dtype=torch.int64, device=device)
    
    print(f"Test data: {n_steps:,} steps, {n_clones.sum().item():,} states")
    
    # Test WITH device checks (slower)
    print("\n=== Testing WITH device checks ===")
    os.environ['CHMM_SKIP_DEVICE_CHECKS'] = '0'  # Enable device checks
    
    start = time.time()
    from models.train_utils import train_chmm
    model1, prog1 = train_chmm(n_clones, x, a, device=device, method='em_T', n_iter=1)
    time_with_checks = time.time() - start
    
    print(f"Time WITH device checks: {time_with_checks:.3f}s")
    
    # Clear module cache to reload with new environment
    if 'models.train_utils' in sys.modules:
        del sys.modules['models.train_utils']
    if 'models.chmm_torch' in sys.modules:
        del sys.modules['models.chmm_torch']
    
    # Test WITHOUT device checks (faster)
    print("\n=== Testing WITHOUT device checks ===")
    os.environ['CHMM_SKIP_DEVICE_CHECKS'] = '1'  # Skip device checks (production mode)
    
    start = time.time()
    from models.train_utils import train_chmm
    model2, prog2 = train_chmm(n_clones, x, a, device=device, method='em_T', n_iter=1)
    time_without_checks = time.time() - start
    
    print(f"Time WITHOUT device checks: {time_without_checks:.3f}s")
    
    # Calculate improvement
    if time_with_checks > 0:
        speedup = time_with_checks / time_without_checks
        improvement_pct = ((time_with_checks - time_without_checks) / time_with_checks) * 100
        
        print(f"\n🚀 PERFORMANCE IMPROVEMENT:")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Time saved: {improvement_pct:.1f}%")
        print(f"   Absolute improvement: {time_with_checks - time_without_checks:.3f}s")
        
        # Extrapolate to 150k steps
        steps_ratio = 150000 / n_steps
        estimated_savings_150k = (time_with_checks - time_without_checks) * steps_ratio
        print(f"\n📊 Estimated savings for 150k steps: {estimated_savings_150k:.1f}s")
        
        if estimated_savings_150k > 60:
            print(f"   That's {estimated_savings_150k/60:.1f} minutes saved!")
    
    # Verify results are identical
    final_bps1 = prog1[-1] if prog1 else 0
    final_bps2 = prog2[-1] if prog2 else 0
    bps_diff = abs(final_bps1 - final_bps2)
    
    print(f"\n✅ Results validation:")
    print(f"   BPS with checks: {final_bps1:.6f}")
    print(f"   BPS without checks: {final_bps2:.6f}")
    print(f"   Difference: {bps_diff:.8f}")
    
    if bps_diff < 1e-6:
        print("   ✅ Results identical - optimization is safe")
    else:
        print("   ⚠️  Results differ - check optimization")
    
    return {
        'time_with_checks': time_with_checks,
        'time_without_checks': time_without_checks,
        'speedup': speedup if time_with_checks > 0 else 1.0,
        'improvement_pct': improvement_pct if time_with_checks > 0 else 0.0
    }

if __name__ == "__main__":
    print("Device Check Optimization Test")
    print("="*40)
    
    results = test_device_optimization()
    
    print(f"\n🎯 Summary:")
    print(f"   Device check optimization provides {results['speedup']:.2f}x speedup")
    print(f"   This should significantly improve A100 performance on 150k steps")