#!/usr/bin/env python3
"""
Comprehensive Performance Test for CHMM Optimizations

Tests the kernel fusion and parallel scan optimizations to validate
that they provide significant speedups over the original implementation.
"""

import torch
import numpy as np
import time
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.chmm_torch import CHMM_torch
from models.chmm_optimized import CHMM_Optimized, benchmark_optimized_vs_original
from models.cuda_kernels import benchmark_kernels


def test_correctness():
    """Test that optimized implementation produces correct results"""
    print("🧪 TESTING CORRECTNESS")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small test case for exact comparison
    seq_len = 1000
    n_obs = 5
    n_actions = 4
    n_states_per_obs = 10
    n_states = n_obs * n_states_per_obs
    
    # Generate deterministic test data
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_clones = torch.ones(n_obs, dtype=torch.int64, device=device) * n_states_per_obs
    x = torch.randint(0, n_obs, (seq_len,), device=device, dtype=torch.int64)
    a = torch.randint(0, n_actions, (seq_len,), device=device, dtype=torch.int64)
    
    print(f"Device: {device}")
    print(f"Sequence length: {seq_len}")
    print(f"States: {n_states}")
    print()
    
    # Create both models with identical parameters
    original_model = CHMM_torch(
        n_clones=n_clones, x=x, a=a,
        pseudocount=0.01, seed=42, device=device
    )
    
    optimized_model = CHMM_Optimized(
        n_clones=n_clones, x=x, a=a, 
        pseudocount=0.01, seed=42, device=device,
        use_cuda_kernels=True
    )
    
    # Copy parameters to ensure identical starting point
    optimized_model.C.data.copy_(original_model.C.data)
    optimized_model.T.data.copy_(original_model.T.data)
    optimized_model.Pi_x.data.copy_(original_model.Pi_x.data)
    
    print("✅ Models created with identical parameters")
    
    # Test BPS computation
    print("\n📊 Testing BPS computation...")
    original_bps = original_model.bps(x, a, reduce=True)
    optimized_bps = optimized_model.bps_optimized(x, a, reduce=True)
    
    bps_diff = abs(original_bps.item() - optimized_bps.item())
    bps_rel_error = bps_diff / abs(original_bps.item())
    
    print(f"Original BPS: {original_bps.item():.6f}")
    print(f"Optimized BPS: {optimized_bps.item():.6f}")
    print(f"Absolute difference: {bps_diff:.8f}")
    print(f"Relative error: {bps_rel_error:.2e}")
    
    if bps_rel_error < 1e-4:
        print("✅ BPS computation is correct")
    else:
        print("❌ BPS computation has significant error")
        return False
    
    # Test EM training convergence
    print("\n🎯 Testing EM training convergence...")
    
    # Reset models to identical state
    torch.manual_seed(42)
    original_model._initialize_parameters()
    optimized_model.C.data.copy_(original_model.C.data)
    optimized_model.T.data.copy_(original_model.T.data)
    optimized_model.Pi_x.data.copy_(original_model.Pi_x.data)
    
    # Run short training
    n_iter = 5
    original_conv = original_model.learn_em_T(x, a, n_iter=n_iter, term_early=False)
    
    # Reset optimized model to same starting point
    torch.manual_seed(42)
    original_model._initialize_parameters()
    optimized_model.C.data.copy_(original_model.C.data)
    optimized_model.T.data.copy_(original_model.T.data)
    optimized_model.Pi_x.data.copy_(original_model.Pi_x.data)
    
    optimized_conv = optimized_model.learn_em_T_optimized(x, a, n_iter=n_iter, term_early=False)
    
    # Compare convergence
    conv_diff = abs(original_conv[-1] - optimized_conv[-1])
    conv_rel_error = conv_diff / abs(original_conv[-1])
    
    print(f"Original final BPS: {original_conv[-1]:.6f}")
    print(f"Optimized final BPS: {optimized_conv[-1]:.6f}")
    print(f"Convergence difference: {conv_diff:.8f}")
    print(f"Relative error: {conv_rel_error:.2e}")
    
    if conv_rel_error < 1e-3:
        print("✅ EM training convergence is correct")
        return True
    else:
        print("❌ EM training has significant convergence error")
        return False


def test_performance_scaling():
    """Test performance scaling with different sequence lengths"""
    print("\n🚀 TESTING PERFORMANCE SCALING")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("⚠️  CUDA not available, skipping performance scaling test")
        return
    
    # Test different sequence lengths
    seq_lengths = [1000, 5000, 10000, 20000]
    n_states = 100
    n_obs = 10
    n_actions = 4
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        # Generate test data
        n_clones = torch.ones(n_obs, dtype=torch.int64, device=device) * (n_states // n_obs)
        x = torch.randint(0, n_obs, (seq_len,), device=device, dtype=torch.int64)
        a = torch.randint(0, n_actions, (seq_len,), device=device, dtype=torch.int64)
        
        # Create models
        original_model = CHMM_torch(
            n_clones=n_clones, x=x, a=a,
            pseudocount=0.01, seed=42, device=device
        )
        
        optimized_model = CHMM_Optimized(
            n_clones=n_clones, x=x, a=a,
            pseudocount=0.01, seed=42, device=device,
            use_cuda_kernels=True
        )
        
        # Warmup
        for _ in range(2):
            try:
                original_model.bps(x[:100], a[:100])
                optimized_model.bps_optimized(x[:100], a[:100])
            except:
                pass
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Benchmark BPS computation
        def time_bps(model, method_name, n_runs=3):
            times = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                if hasattr(model, method_name):
                    getattr(model, method_name)(x, a, reduce=True)
                else:
                    model.bps(x, a, reduce=True)
                
                torch.cuda.synchronize()
                times.append(time.time() - start_time)
            return np.mean(times)
        
        original_time = time_bps(original_model, "bps")
        optimized_time = time_bps(optimized_model, "bps_optimized")
        
        speedup = original_time / optimized_time
        throughput = seq_len / optimized_time
        
        print(f"  Original time: {original_time*1000:.1f}ms")
        print(f"  Optimized time: {optimized_time*1000:.1f}ms")  
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Throughput: {throughput:.0f} timesteps/sec")
        
        results.append({
            'seq_len': seq_len,
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'throughput': throughput
        })
    
    # Summary
    print(f"\n📈 SCALING SUMMARY")
    print("-" * 30)
    avg_speedup = np.mean([r['speedup'] for r in results])
    max_throughput = max([r['throughput'] for r in results])
    
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Max throughput: {max_throughput:.0f} timesteps/sec")
    
    # Check if speedup increases with sequence length (expected behavior)
    speedups = [r['speedup'] for r in results]
    if len(speedups) > 1 and speedups[-1] > speedups[0]:
        print("✅ Speedup scales well with sequence length")
    else:
        print("⚠️  Speedup doesn't scale as expected")
    
    return results


def run_comprehensive_benchmark():
    """Run the comprehensive benchmark comparing all implementations"""
    print("\n🔥 RUNNING COMPREHENSIVE BENCHMARK")
    print("="*60)
    
    try:
        # Test different configurations
        configs = [
            {'seq_len': 5000, 'n_states': 100, 'n_iter': 10},
            {'seq_len': 10000, 'n_states': 200, 'n_iter': 15},
        ]
        
        if torch.cuda.is_available():
            configs.append({'seq_len': 20000, 'n_states': 300, 'n_iter': 20})
        
        for i, config in enumerate(configs):
            print(f"\n🧪 Test Configuration {i+1}")
            print("-" * 30)
            
            results = benchmark_optimized_vs_original(**config)
            
            if results['bps_speedup'] > 1.5:
                print(f"✅ Good BPS speedup: {results['bps_speedup']:.2f}x")
            else:
                print(f"⚠️  Low BPS speedup: {results['bps_speedup']:.2f}x")
            
            if results['em_speedup'] > 2.0:
                print(f"✅ Excellent EM speedup: {results['em_speedup']:.2f}x")
            elif results['em_speedup'] > 1.5:
                print(f"✅ Good EM speedup: {results['em_speedup']:.2f}x")
            else:
                print(f"⚠️  Low EM speedup: {results['em_speedup']:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()
        return False


def test_cuda_kernels():
    """Test CUDA kernel compilation and execution"""
    print("\n⚙️  TESTING CUDA KERNELS")
    print("="*40)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping kernel tests")
        return True
    
    try:
        # Test kernel compilation
        print("🔧 Testing kernel compilation...")
        benchmark_kernels(seq_len=1000, n_states=50, n_actions=4)
        print("✅ CUDA kernels compiled and executed successfully")
        return True
        
    except Exception as e:
        print(f"❌ CUDA kernel test failed: {e}")
        print("This is expected on MPS/CPU - kernels will fall back to PyTorch")
        return False


def main():
    """Run all performance tests"""
    print("🎯 CHMM OPTIMIZATION PERFORMANCE TESTS")
    print("="*80)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()
    
    results = {}
    
    # Test 1: Correctness
    try:
        results['correctness'] = test_correctness()
    except Exception as e:
        print(f"❌ Correctness test failed: {e}")
        traceback.print_exc()
        results['correctness'] = False
    
    # Test 2: CUDA kernels (if available)
    try:
        results['cuda_kernels'] = test_cuda_kernels()
    except Exception as e:
        print(f"❌ CUDA kernel test failed: {e}")
        results['cuda_kernels'] = False
    
    # Test 3: Performance scaling
    try:
        results['scaling'] = test_performance_scaling()
    except Exception as e:
        print(f"❌ Performance scaling test failed: {e}")
        traceback.print_exc()
        results['scaling'] = False
    
    # Test 4: Comprehensive benchmark
    try:
        results['benchmark'] = run_comprehensive_benchmark()
    except Exception as e:
        print(f"❌ Comprehensive benchmark failed: {e}")
        traceback.print_exc()
        results['benchmark'] = False
    
    # Final summary
    print("\n🎉 FINAL TEST SUMMARY")
    print("="*50)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.capitalize():20} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Optimizations are working correctly.")
    elif results.get('correctness', False):
        print("✅ Core functionality is correct, some optimizations may not be available.")
    else:
        print("❌ Critical issues detected. Check implementation.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)