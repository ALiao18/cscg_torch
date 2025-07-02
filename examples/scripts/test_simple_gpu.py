#!/usr/bin/env python3
"""
Simple test for GPU sequence generation
"""
import torch
import numpy as np
import time
from env_adapters.room_utils import create_room_adapter

def simple_test():
    """Simple test of GPU sequence generation"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps:0")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print(f"Testing on device: {device}")
    
    # Create simple 3x3 room
    room_data = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    
    room_tensor = torch.tensor(room_data, device=device, dtype=torch.int64)
    adapter = create_room_adapter(room_tensor, adapter_type="torch", seed=42)
    
    # Test very small sequence first
    print("Testing small sequence (100 steps)...")
    try:
        if hasattr(adapter, 'generate_sequence_gpu'):
            start_time = time.time()
            x_gpu, a_gpu = adapter.generate_sequence_gpu(100, device)
            gpu_time = time.time() - start_time
            print(f"GPU generation: {gpu_time:.4f} seconds")
            print(f"Generated {len(x_gpu)} observations, {len(a_gpu)} actions")
            print(f"Obs range: [{x_gpu.min()}, {x_gpu.max()}]")
            print(f"Action range: [{a_gpu.min()}, {a_gpu.max()}]")
        else:
            print("GPU generation method not available")
            
        # Test CPU version for comparison
        start_time = time.time()
        x_cpu, a_cpu = adapter.generate_sequence(100)
        cpu_time = time.time() - start_time
        print(f"CPU generation: {cpu_time:.4f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()