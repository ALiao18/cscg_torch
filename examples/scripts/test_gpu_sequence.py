#!/usr/bin/env python3
"""
Test GPU-accelerated sequence generation performance
"""
import torch
import numpy as np
import time
from env_adapters.room_utils import create_room_adapter, generate_room_sequence

def test_sequence_generation_performance():
    """Compare CPU vs GPU sequence generation performance"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps:0")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print(f"Testing on device: {device}")
    
    # Create test room
    room_data = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 0, 1, 2, 3],
        [4, 5, 6, 7, 8]
    ])
    
    room_tensor = torch.tensor(room_data, device=device, dtype=torch.int64)
    adapter = create_room_adapter(room_tensor, adapter_type="torch", seed=42)
    
    # Test different sequence lengths
    test_lengths = [1000, 5000, 10000, 25000]
    
    print("\n=== Sequence Generation Performance Test ===")
    print(f"{'Length':<8} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10} {'Status'}")
    print("-" * 55)
    
    for seq_len in test_lengths:
        try:
            # Test CPU generation
            start_time = time.time()
            x_cpu, a_cpu = generate_room_sequence(adapter, seq_len, use_gpu=False)
            cpu_time = time.time() - start_time
            
            # Test GPU generation
            start_time = time.time()
            x_gpu, a_gpu = generate_room_sequence(adapter, seq_len, use_gpu=True, device=device)
            gpu_time = time.time() - start_time
            
            # Calculate speedup
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            
            # Verify results are consistent
            if np.array_equal(x_cpu, x_gpu) and np.array_equal(a_cpu, a_gpu):
                status = "✓ Match"
            else:
                status = "✗ Differ"
            
            print(f"{seq_len:<8} {cpu_time:<12.4f} {gpu_time:<12.4f} {speedup:<10.2f}x {status}")
            
        except Exception as e:
            print(f"{seq_len:<8} {'Error:':<25} {str(e)}")
    
    print()
    
    # Test very long sequence if GPU is available
    if device.type != 'cpu':
        print("=== Long Sequence Test (GPU only) ===")
        long_lengths = [50000, 100000]
        
        for seq_len in long_lengths:
            try:
                print(f"Generating {seq_len} steps on GPU...")
                start_time = time.time()
                x_long, a_long = generate_room_sequence(adapter, seq_len, use_gpu=True, device=device)
                gpu_time = time.time() - start_time
                
                print(f"  Time: {gpu_time:.4f} seconds")
                print(f"  Rate: {seq_len/gpu_time:.0f} steps/second")
                print(f"  Sequence shape: x={x_long.shape}, a={a_long.shape}")
                
                # Basic validation
                assert len(x_long) == seq_len, f"Length mismatch: {len(x_long)} != {seq_len}"
                assert len(a_long) == seq_len, f"Length mismatch: {len(a_long)} != {seq_len}"
                assert np.all(x_long >= 0) and np.all(x_long <= 15), "Invalid observations"
                assert np.all(a_long >= 0) and np.all(a_long <= 3), "Invalid actions"
                
                print(f"  Validation: ✓ Passed")
                
            except Exception as e:
                print(f"  Error: {e}")
            
            print()

if __name__ == "__main__":
    test_sequence_generation_performance()