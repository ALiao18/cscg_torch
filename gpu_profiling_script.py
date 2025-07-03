#!/usr/bin/env python3
"""
GPU Profiling Script for CHMM Training
Identifies bottlenecks in matrix operations and GPU utilization
"""

import torch
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import psutil
import subprocess

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0

def profile_matrix_operations():
    """Profile core matrix operations used in CHMM training"""
    print("=== Profiling Matrix Operations ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different matrix sizes
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\n--- Testing size {size}x{size} ---")
        
        # Create test matrices
        A = torch.randn(size, size, device=device, dtype=torch.float32)
        B = torch.randn(size, size, device=device, dtype=torch.float32)
        v = torch.randn(size, device=device, dtype=torch.float32)
        
        # Time different operations
        operations = {
            'matmul': lambda: torch.matmul(A, B),
            'mv': lambda: torch.mv(A, v),  # matrix-vector multiplication
            'bmm': lambda: torch.bmm(A.unsqueeze(0), B.unsqueeze(0)),  # batch matrix multiply
            'sum': lambda: A.sum(dim=1),
            'div': lambda: A / A.sum(dim=1, keepdim=True),
            'log': lambda: torch.log(A.abs() + 1e-8),
            'cumsum': lambda: torch.cumsum(A, dim=0)
        }
        
        for op_name, op_func in operations.items():
            # Warmup
            for _ in range(3):
                _ = op_func()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Time the operation
            start_time = time.time()
            for _ in range(10):
                result = op_func()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            mem_alloc, mem_reserved = get_gpu_memory()
            
            print(f"  {op_name:8s}: {avg_time*1000:.2f}ms, GPU mem: {mem_alloc:.2f}GB/{mem_reserved:.2f}GB")
        
        # Clean up
        del A, B, v, result
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

def profile_chmm_forward_pass():
    """Profile the actual CHMM forward pass operations"""
    print("\n=== Profiling CHMM Forward Pass ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate CHMM parameters
    n_states = 100
    n_actions = 4
    seq_length = 1000
    
    print(f"CHMM params: {n_states} states, {n_actions} actions, {seq_length} sequence length")
    
    # Create transition matrix (transposed for efficiency)
    T_tr = torch.rand(n_actions, n_states, n_states, device=device, dtype=torch.float32)
    T_tr = T_tr / T_tr.sum(dim=2, keepdim=True)  # Normalize
    
    # Create initial distribution
    Pi = torch.rand(n_states, device=device, dtype=torch.float32)
    Pi = Pi / Pi.sum()
    
    # Create observation and action sequences
    x = torch.randint(0, 10, (seq_length,), device=device)
    a = torch.randint(0, n_actions, (seq_length,), device=device)
    n_clones = torch.tensor([10] * 10, device=device)  # 10 clones per observation
    
    print("Starting forward pass profiling...")
    
    # Profile the forward pass
    def simulate_forward_pass():
        state_loc = torch.cat([torch.tensor([0], device=device), n_clones]).cumsum(0)
        log2_lik = torch.zeros(seq_length, device=device)
        
        # Initialize message at t=0
        t = 0
        j = x[t]
        j_start, j_stop = state_loc[j : j + 2]
        message = Pi[j_start:j_stop].clone()
        p_obs = message.sum()
        message = message / p_obs
        log2_lik[0] = torch.log2(p_obs)
        
        # Forward pass loop
        for t in range(1, seq_length):
            ajt = a[t - 1]
            i, j = x[t - 1], x[t]
            
            i_start, i_stop = state_loc[i : i + 2]
            j_start, j_stop = state_loc[j : j + 2]
            
            # Critical matrix-vector multiplication
            with record_function("matrix_vector_mul"):
                T_tr_slice = T_tr[ajt, j_start:j_stop, i_start:i_stop]
                message = torch.matmul(T_tr_slice, message)
            
            with record_function("normalization"):
                p_obs = message.sum()
                message = message / p_obs
                log2_lik[t] = torch.log2(p_obs)
        
        return log2_lik
    
    # Warmup
    for _ in range(3):
        _ = simulate_forward_pass()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Profile with PyTorch profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        result = simulate_forward_pass()
    
    # Print profiling results
    print("\nTop time-consuming operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=10))
    
    print("\nMemory usage by operation:")
    print(prof.key_averages().table(sort_by="cuda_memory_usage" if torch.cuda.is_available() else "cpu_memory_usage", row_limit=10))
    
    # Time multiple runs
    times = []
    for _ in range(10):
        start_time = time.time()
        _ = simulate_forward_pass()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        times.append(end_time - start_time)
    
    print(f"\nForward pass timing (10 runs):")
    print(f"  Mean: {np.mean(times)*1000:.2f}ms")
    print(f"  Std:  {np.std(times)*1000:.2f}ms")
    print(f"  Min:  {np.min(times)*1000:.2f}ms")
    print(f"  Max:  {np.max(times)*1000:.2f}ms")

def check_gpu_utilization():
    """Check GPU utilization using nvidia-smi"""
    print("\n=== GPU Utilization Check ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU utilization check")
        return
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                util, mem_used, mem_total, temp = line.split(', ')
                print(f"GPU {i}: {util}% utilization, {mem_used}/{mem_total}MB memory, {temp}°C")
        else:
            print("Failed to get GPU info from nvidia-smi")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("nvidia-smi not available")
    
    # PyTorch GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"PyTorch GPU {i}: {props.name}, {props.total_memory/1024**3:.1f}GB total memory")
            print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB")
            print(f"  Reserved:  {torch.cuda.memory_reserved(i)/1024**3:.2f}GB")

def test_data_transfer_overhead():
    """Test CPU-GPU data transfer overhead"""
    print("\n=== Data Transfer Overhead Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping data transfer test")
        return
    
    sizes = [100, 1000, 10000, 100000]
    
    for size in sizes:
        print(f"\n--- Testing {size}x{size} tensor transfer ---")
        
        # Create tensor on CPU
        cpu_tensor = torch.randn(size, size, dtype=torch.float32)
        
        # Time CPU to GPU transfer
        start_time = time.time()
        gpu_tensor = cpu_tensor.to('cuda')
        torch.cuda.synchronize()
        cpu_to_gpu_time = time.time() - start_time
        
        # Time GPU to CPU transfer
        start_time = time.time()
        back_to_cpu = gpu_tensor.to('cpu')
        torch.cuda.synchronize()
        gpu_to_cpu_time = time.time() - start_time
        
        tensor_size_mb = cpu_tensor.numel() * 4 / 1024**2  # 4 bytes per float32
        
        print(f"  Tensor size: {tensor_size_mb:.1f}MB")
        print(f"  CPU->GPU: {cpu_to_gpu_time*1000:.2f}ms ({tensor_size_mb/cpu_to_gpu_time:.1f}MB/s)")
        print(f"  GPU->CPU: {gpu_to_cpu_time*1000:.2f}ms ({tensor_size_mb/gpu_to_cpu_time:.1f}MB/s)")
        
        del cpu_tensor, gpu_tensor, back_to_cpu
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Starting GPU Profiling for CHMM Training")
    print("=" * 50)
    
    # System info
    print(f"Python executable: {psutil.Process().exe()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU devices: {torch.cuda.device_count()}")
    
    # Run profiling tests
    try:
        check_gpu_utilization()
        profile_matrix_operations()
        test_data_transfer_overhead()
        profile_chmm_forward_pass()
        
        print("\n" + "=" * 50)
        print("Profiling completed successfully!")
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()