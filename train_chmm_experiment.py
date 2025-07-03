#!/usr/bin/env python3
"""
Small-scale experiment to identify GPU optimization bottlenecks in train_chmm
"""

import torch
import time
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.train_utils import train_chmm

def create_test_data(n_observations=10, n_actions=4, seq_length=100, device='cuda'):
    """Create synthetic test data for CHMM training"""
    # Number of clones per observation (states)
    n_clones = torch.randint(5, 15, (n_observations,), dtype=torch.int64)
    total_states = n_clones.sum().item()
    
    # Random observation sequence
    x = torch.randint(0, n_observations, (seq_length,), dtype=torch.int64)
    
    # Random action sequence  
    a = torch.randint(0, n_actions, (seq_length,), dtype=torch.int64)
    
    print(f"Test data: {n_observations}obs, {seq_length}seq, {total_states}states")
    
    return n_clones, x, a

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

def time_function(func, description, *args, **kwargs):
    """Time a function with GPU synchronization"""
    
    # Silent warmup
    try:
        _ = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return None, None
    
    # Timed run
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.time() - start_time
        
        print(f"{description}: {duration:.3f}s")
        return result, duration
        
    except Exception as e:
        print(f"{description}: FAILED - {e}")
        return None, None

def experiment_different_sizes():
    """Test train_chmm with different problem sizes"""
    print("\n=== Experimenting with Different Problem Sizes ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test configurations: (n_obs, seq_length, n_iter)
    configs = [
        (5, 50, 5),      # Small
        (10, 100, 5),    # Medium
        (20, 200, 5),    # Large
        (10, 500, 3),    # Long sequence
    ]
    
    results = []
    
    for n_obs, seq_len, n_iter in configs:
        print(f"\n{'='*50}")
        print(f"Configuration: {n_obs} observations, {seq_len} sequence length, {n_iter} iterations")
        
        # Create test data
        n_clones, x, a = create_test_data(n_obs, 4, seq_len, device)
        
        # Run training with timing
        def run_training():
            return train_chmm(
                n_clones=n_clones,
                x=x,
                a=a,
                device=device,
                method='em_T',
                n_iter=n_iter,
                pseudocount=0.01,
                seed=42,
                early_stopping=False,  # Disable for consistent timing
                use_mixed_precision=False,  # Test without mixed precision first
                memory_efficient=True
            )
        
        result, duration = time_function(
            run_training,
            f"train_chmm({n_obs}obs, {seq_len}seq, {n_iter}iter)"
        )
        
        if result is not None:
            model, progression = result
            total_states = n_clones.sum().item()
            
            results.append({
                'n_obs': n_obs,
                'seq_len': seq_len,
                'n_iter': n_iter,
                'total_states': total_states,
                'duration': duration,
                'final_bps': progression[-1] if progression else None,
                'improvement': progression[0] - progression[-1] if len(progression) > 1 else None
            })
            
            print(f"Final BPS: {progression[-1]:.4f}")
            if len(progression) > 1:
                print(f"Total improvement: {progression[0] - progression[-1]:.4f}")
        
        # Clean up GPU memory
        del n_clones, x, a
        if result is not None:
            del model, progression
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"{'Config':<20} {'States':<8} {'Duration':<10} {'BPS':<8} {'Improvement':<12}")
    print("-" * 70)
    
    for r in results:
        config_str = f"{r['n_obs']}x{r['seq_len']}x{r['n_iter']}"
        duration_str = f"{r['duration']:.3f}s" if r['duration'] else "FAILED"
        bps_str = f"{r['final_bps']:.4f}" if r['final_bps'] else "N/A"
        imp_str = f"{r['improvement']:.4f}" if r['improvement'] else "N/A"
        
        print(f"{config_str:<20} {r['total_states']:<8} {duration_str:<10} {bps_str:<8} {imp_str:<12}")

def test_optimization_improvements():
    """Test the impact of our GPU optimizations"""
    print("\n=== Testing Optimization Improvements ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fixed test configuration  
    n_clones, x, a = create_test_data(10, 4, 200, device)
    
    optimization_configs = [
        ("Standard", {"use_mixed_precision": False, "memory_efficient": False}),
        ("Memory Efficient", {"use_mixed_precision": False, "memory_efficient": True}),
        ("Mixed Precision", {"use_mixed_precision": True, "memory_efficient": False}),
        ("Fully Optimized", {"use_mixed_precision": True, "memory_efficient": True}),
    ]
    
    results = {}
    
    for config_name, config_params in optimization_configs:
        print(f"\n--- Testing {config_name} Configuration ---")
        
        def run_optimized():
            return train_chmm(
                n_clones=n_clones, x=x, a=a, device=device,
                method='em_T', n_iter=5, pseudocount=0.01, seed=42,
                early_stopping=False, **config_params
            )
        
        result, duration = time_function(run_optimized, f"{config_name} training")
        
        if result is not None:
            model, progression = result
            results[config_name] = {
                'duration': duration,
                'final_bps': progression[-1],
                'improvement': progression[0] - progression[-1] if len(progression) > 1 else 0
            }
    
    # Compare optimization results
    print(f"\n{'='*60}")
    print("OPTIMIZATION IMPACT ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Configuration':<15} {'Duration':<10} {'Speedup':<10} {'Final BPS':<12}")
    print("-" * 60)
    
    baseline_duration = results.get("Standard", {}).get('duration', 1.0)
    
    for config_name, result_data in results.items():
        speedup = baseline_duration / result_data['duration'] if result_data['duration'] > 0 else 0
        print(f"{config_name:<15} {result_data['duration']:.3f}s    {speedup:.2f}x      {result_data['final_bps']:.4f}")

def analyze_training_methods():
    """Compare different training methods"""
    print("\n=== Analyzing Different Training Methods ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fixed test configuration
    n_clones, x, a = create_test_data(10, 4, 100, device)
    
    methods = ['em_T', 'viterbi_T']
    method_results = {}
    
    for method in methods:
        print(f"\n--- Testing method: {method} ---")
        
        def run_method():
            return train_chmm(
                n_clones=n_clones, x=x, a=a, device=device,
                method=method, n_iter=10, pseudocount=0.01, seed=42,
                early_stopping=False, use_mixed_precision=False, memory_efficient=True
            )
        
        result, duration = time_function(run_method, f"train_chmm with {method}")
        
        if result is not None:
            model, progression = result
            method_results[method] = {
                'duration': duration,
                'final_bps': progression[-1],
                'improvement': progression[0] - progression[-1] if len(progression) > 1 else 0
            }
    
    # Compare results
    print(f"\n{'='*40}")
    print("METHOD COMPARISON")
    print(f"{'='*40}")
    print(f"{'Method':<12} {'Duration':<10} {'Final BPS':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for method, results in method_results.items():
        print(f"{method:<12} {results['duration']:.3f}s    {results['final_bps']:.4f}     {results['improvement']:.4f}")

def profile_granular_em_operations():
    """Profile granular operations within EM training to find remaining bottlenecks"""
    print("\n=== Granular EM Operations Profiling ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data with different sizes to test scaling
    test_configs = [
        (10, 100, "Small"),
        (15, 200, "Medium"), 
        (20, 300, "Large")
    ]
    
    for n_obs, seq_len, size_label in test_configs:
        print(f"\n--- {size_label} Problem Size: {n_obs} obs, {seq_len} seq ---")
        
        n_clones, x, a = create_test_data(n_obs, 4, seq_len, device)
        
        # Initialize model once
        from models.chmm_torch import CHMM_torch
        model = CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
        
        print(f"Total states: {model.n_states}, Sequence length: {seq_len}")
        
        # Profile core EM components
        profile_forward_backward_operations(model, n_clones, x, a, device)
        profile_parameter_update_operations(model, n_clones, x, a, device)
        profile_tensor_operations(model, n_clones, x, a, device, seq_len)
        
        # Clean up
        del model, n_clones, x, a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def profile_forward_backward_operations(model, n_clones, x, a, device):
    """Profile forward and backward pass operations separately"""
    from models.train_utils import forward, backward
    
    # 1. Forward pass without message storage
    def forward_no_messages():
        return forward(
            T_tr=model.T.transpose(1, 2),
            Pi=model.Pi_x,
            n_clones=n_clones,
            x=x, a=a, device=device,
            store_messages=False
        )
    
    # 2. Forward pass with message storage  
    def forward_with_messages():
        return forward(
            T_tr=model.T.transpose(1, 2),
            Pi=model.Pi_x, 
            n_clones=n_clones,
            x=x, a=a, device=device,
            store_messages=True
        )
    
    # 3. Backward pass
    def backward_pass():
        return backward(
            T=model.T,
            n_clones=n_clones,
            x=x, a=a, device=device
        )
    
    # 4. Matrix transpose operation
    def transpose_operation():
        return model.T.transpose(1, 2)
    
    time_function(forward_no_messages, "Forward pass (no messages)")
    time_function(forward_with_messages, "Forward pass (with messages)")  
    time_function(backward_pass, "Backward pass")
    time_function(transpose_operation, "Transition matrix transpose")

def profile_parameter_update_operations(model, n_clones, x, a, device):
    """Profile parameter update operations"""
    from models.train_utils import forward, backward, updateC
    
    # Get forward and backward messages first
    log2_lik, mess_fwd = forward(
        T_tr=model.T.transpose(1, 2), Pi=model.Pi_x,
        n_clones=n_clones, x=x, a=a, device=device, store_messages=True
    )
    mess_bwd = backward(T=model.T, n_clones=n_clones, x=x, a=a, device=device)
    
    # 1. Count matrix update (the expensive operation)
    def update_counts():
        C = torch.zeros_like(model.T)
        updateC(C, model.T, n_clones, mess_fwd, mess_bwd, x, a, device)
        return C
    
    # 2. Transition matrix normalization
    def normalize_transition():
        C = torch.rand_like(model.T) + 0.1  # Simulate count matrix
        T_new = C / (C.sum(dim=2, keepdim=True) + 1e-10)
        return T_new
    
    # 3. BPS computation  
    def compute_bps():
        return log2_lik.sum().item() / len(x)
    
    time_function(update_counts, "Count matrix update (updateC)")
    time_function(normalize_transition, "Transition matrix normalization")
    time_function(compute_bps, "BPS computation")

def profile_tensor_operations(model, n_clones, x, a, device, seq_len):
    """Profile low-level tensor operations for bottlenecks"""
    
    # 1. State location computation
    def compute_state_locations():
        return torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones.cumsum(0)])
    
    # 2. Message location computation  
    def compute_message_locations():
        return torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones[x].cumsum(0)])
    
    # 3. Tensor indexing operations
    def tensor_indexing_ops():
        state_loc = compute_state_locations()
        total_ops = 0
        for t in range(1, min(50, seq_len)):  # Sample first 50 timesteps
            i = x[t-1].item()
            j = x[t].item() 
            ajt = a[t-1].item()
            
            i_start, i_stop = state_loc[i:i+2]
            j_start, j_stop = state_loc[j:j+2]
            
            # This is the expensive slice operation
            T_slice = model.T[ajt, i_start:i_stop, j_start:j_stop]
            total_ops += T_slice.numel()
        return total_ops
    
    # 4. Matrix-vector multiplication benchmark
    def benchmark_mv_operations():
        state_loc = compute_state_locations()
        total_time = 0
        n_ops = min(20, seq_len - 1)
        
        for t in range(1, n_ops + 1):
            i = x[t-1].item()
            j = x[t].item()
            ajt = a[t-1].item()
            
            i_start, i_stop = state_loc[i:i+2]
            j_start, j_stop = state_loc[j:j+2] 
            
            T_slice = model.T[ajt, j_start:j_stop, i_start:i_stop]
            message = torch.randn(i_stop - i_start, device=model.T.device, dtype=model.T.dtype)
            
            start = time.time()
            result = torch.mv(T_slice, message)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time += time.time() - start
            
        return total_time / n_ops if n_ops > 0 else 0
    
    time_function(compute_state_locations, "State location computation")
    time_function(compute_message_locations, "Message location computation") 
    time_function(tensor_indexing_ops, "Tensor indexing operations")
    
    # Benchmark matrix-vector ops
    avg_mv_time = benchmark_mv_operations()
    print(f"Average matrix-vector multiply time: {avg_mv_time*1000:.3f}ms")

def profile_memory_access_patterns():
    """Profile memory access patterns and allocation overhead"""
    print("\n=== Memory Access Pattern Analysis ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different memory allocation patterns
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\n--- Testing memory patterns for size {size} ---")
        
        # 1. Sequential tensor creation
        def sequential_allocation():
            tensors = []
            for i in range(10):
                tensor = torch.randn(size, size, device=device)
                tensors.append(tensor)
            return tensors
        
        # 2. Batch tensor creation
        def batch_allocation():
            return torch.randn(10, size, size, device=device)
        
        # 3. In-place operations vs new tensor creation
        def inplace_operations():
            tensor = torch.randn(size, size, device=device)
            for _ in range(10):
                tensor.add_(0.1)
            return tensor
        
        def new_tensor_operations():
            tensor = torch.randn(size, size, device=device)
            for _ in range(10):
                tensor = tensor + 0.1
            return tensor
        
        time_function(sequential_allocation, f"Sequential allocation ({size}x{size})")
        time_function(batch_allocation, f"Batch allocation ({size}x{size})")
        time_function(inplace_operations, f"In-place operations ({size}x{size})")
        time_function(new_tensor_operations, f"New tensor operations ({size}x{size})")

def profile_individual_operations():
    """Profile individual operations within train_chmm"""
    print("\n=== Individual Operations Profiling ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    n_clones, x, a = create_test_data(15, 4, 200, device)
    
    print("Testing basic operations:")
    
    # 1. Device movement
    cpu_n_clones = n_clones.cpu()
    cpu_x = x.cpu() 
    cpu_a = a.cpu()
    
    def move_to_device():
        return cpu_n_clones.to(device), cpu_x.to(device), cpu_a.to(device)
    
    time_function(move_to_device, "Moving tensors to device")
    
    # 2. Model initialization
    def init_model():
        from models.chmm_torch import CHMM_torch
        return CHMM_torch(
            n_clones=n_clones, x=x, a=a,
            pseudocount=0.01, seed=42
        )
    
    model, _ = time_function(init_model, "CHMM model initialization")
    
    if model is not None:
        # 3. Single EM step  
        def single_em_step():
            return model.learn_em_T(x=x, a=a, n_iter=1, term_early=False)
        
        time_function(single_em_step, "Single EM iteration")
        
        # Run granular profiling
        profile_granular_em_operations()
        profile_memory_access_patterns()

if __name__ == "__main__":
    print("Starting train_chmm GPU Optimization Analysis")
    print("=" * 60)
    
    # System info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # Run experiments
        experiment_different_sizes()
        test_optimization_improvements()
        analyze_training_methods()
        profile_individual_operations()
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        
        # Final GPU memory status
        if torch.cuda.is_available():
            final_mem = monitor_gpu_memory()
            print(f"Final GPU memory: {final_mem[0]:.2f}GB allocated, {final_mem[1]:.2f}GB reserved")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()