#!/usr/bin/env python3
"""
Test GPU sequence generation with longer sequences
"""
import torch
import numpy as np
import time
from env_adapters.room_utils import create_room_adapter

def test_long_sequences():
    """Test GPU sequence generation with various lengths"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps:0")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print(f"Testing long sequences on device: {device}")
    
    # Create room from training script
    room_data = np.array([
        [ 6, 10,  7,  6, 15, 12, 13, 10,  2,  8,  3,  2,  2,  8,  0, 10, 14, 14, 11, 11],
        [ 7, 15, 10,  3,  8,  1,  7,  6,  3,  9, 15, 12,  3,  9, 14,  8, 10,  1,  1,  4],
        [13,  8,  6, 13,  3,  9, 12,  4,  9, 12,  3,  3, 13,  4,  3, 11, 10, 11,  7,  3],
        [ 9,  1,  6,  7, 14, 15,  3,  8,  0,  8,  6, 13, 12,  5,  4,  1,  6, 13,  5,  5],
        [13, 13,  8,  1,  8,  8,  5,  4, 15,  8,  7,  7,  7,  0,  1,  4,  5, 11, 10,  1],
    ])
    
    room_tensor = torch.tensor(room_data, device=device, dtype=torch.int64)
    adapter = create_room_adapter(room_tensor, adapter_type="torch", seed=42)
    
    # Test different sequence lengths
    test_lengths = [10000, 25000, 50000, 100000]
    
    print("\n=== GPU Sequence Generation Performance ===")
    print(f"{'Length':<8} {'Time (s)':<10} {'Rate (steps/s)':<15} {'Status'}")
    print("-" * 50)
    
    for seq_len in test_lengths:
        try:
            print(f"\nTesting {seq_len} steps...")
            start_time = time.time()
            
            x_seq, a_seq = adapter.generate_sequence_gpu(seq_len, device)
            
            end_time = time.time()
            generation_time = end_time - start_time
            rate = seq_len / generation_time
            
            # Basic validation
            assert len(x_seq) == seq_len, f"Length mismatch: {len(x_seq)} != {seq_len}"
            assert len(a_seq) == seq_len, f"Length mismatch: {len(a_seq)} != {seq_len}"
            assert np.all(x_seq >= 0) and np.all(x_seq <= 15), "Invalid observations"
            assert np.all(a_seq >= 0) and np.all(a_seq <= 3), "Invalid actions"
            
            status = "✓ Pass"
            print(f"{seq_len:<8} {generation_time:<10.3f} {rate:<15.0f} {status}")
            
        except Exception as e:
            print(f"{seq_len:<8} {'Error:':<25} {str(e)[:30]}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    test_long_sequences()