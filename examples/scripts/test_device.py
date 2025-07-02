#!/usr/bin/env python3
"""
Quick device test to check MPS functionality
"""
import torch
import time

def test_device_performance():
    # Test device availability
    print("=== Device Availability ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Selected device: {device}")
    
    # Test basic tensor operations
    print("\n=== Basic Tensor Test ===")
    try:
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        start_time = time.time()
        c = torch.matmul(a, b)
        end_time = time.time()
        
        print(f"Matrix multiplication successful on {c.device}")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        print(f"Result shape: {c.shape}")
        
    except Exception as e:
        print(f"Basic tensor test failed: {e}")
    
    # Test tensor conversion and movement
    print("\n=== Tensor Movement Test ===")
    try:
        cpu_tensor = torch.randn(100, 100)
        print(f"CPU tensor device: {cpu_tensor.device}")
        
        gpu_tensor = cpu_tensor.to(device)
        print(f"GPU tensor device: {gpu_tensor.device}")
        
        back_to_cpu = gpu_tensor.cpu()
        print(f"Back to CPU device: {back_to_cpu.device}")
        
    except Exception as e:
        print(f"Tensor movement test failed: {e}")
        
    # Test sequence generation performance
    print("\n=== Sequence Generation Performance ===")
    try:
        # Simulate sequence generation with random choices
        seq_len = 15000
        n_actions = 4
        
        start_time = time.time()
        
        # CPU-based (current approach)
        import numpy as np
        rng = np.random.RandomState(42)
        cpu_actions = [rng.choice(n_actions) for _ in range(seq_len)]
        
        cpu_time = time.time() - start_time
        print(f"CPU sequence generation time: {cpu_time:.4f} seconds")
        
        # GPU-based alternative
        start_time = time.time()
        gpu_actions = torch.randint(0, n_actions, (seq_len,), device=device)
        gpu_time = time.time() - start_time
        
        print(f"GPU sequence generation time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"Sequence generation test failed: {e}")

if __name__ == "__main__":
    test_device_performance()