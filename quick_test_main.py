#!/usr/bin/env python3
"""
Quick Test: Focus on the main objective - env.generate_sequence(150000)

This test specifically validates that we can generate large sequences 
without CUDA tensor conversion errors.
"""

import torch
import numpy as np
import time
import traceback

def create_test_room(size=50):
    """Create a test room with walls and open spaces."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create room with walls on edges and some internal structure
    room = torch.randint(0, 3, size=[size, size], dtype=torch.long, device=device)
    
    # Add boundary walls
    room[0, :] = -1  # Top wall
    room[-1, :] = -1  # Bottom wall
    room[:, 0] = -1  # Left wall
    room[:, -1] = -1  # Right wall
    
    # Ensure some free spaces exist
    room[1:size-1, 1:size-1] = torch.where(
        room[1:size-1, 1:size-1] == -1, 
        torch.randint(0, 2, size=[size-2, size-2], dtype=torch.long, device=device),
        room[1:size-1, 1:size-1]
    )
    
    return room

def main():
    """Test the main objective: 150k sequence generation without CUDA errors."""
    print("🧪 QUICK TEST: env.generate_sequence(150000)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    try:
        from env_adapters.room_adapter import RoomTorchAdapter, RoomNPAdapter
        print("✅ Imports successful")
        
        # Test with PyTorch adapter (most likely to have CUDA issues)
        print("\n📍 Testing RoomTorchAdapter...")
        room_tensor = create_test_room(size=50)
        torch_adapter = RoomTorchAdapter(room_tensor, seed=42)
        
        print("🎯 MAIN TEST: Generating 150,000 steps...")
        start_time = time.time()
        
        x_seq, a_seq = torch_adapter.generate_sequence(150000)
        
        elapsed = time.time() - start_time
        
        # Critical validations
        print(f"✅ SUCCESS! Generated {len(x_seq)} steps in {elapsed:.2f}s")
        print(f"✅ Return types: x={type(x_seq)}, a={type(a_seq)}")
        print(f"✅ Data types: x={x_seq.dtype}, a={a_seq.dtype}")
        print(f"✅ Observation range: [{x_seq.min()}, {x_seq.max()}]")
        print(f"✅ Action range: [{a_seq.min()}, {a_seq.max()}]")
        
        # Verify no CUDA tensors returned
        assert isinstance(x_seq, np.ndarray), f"x_seq should be numpy array, got {type(x_seq)}"
        assert isinstance(a_seq, np.ndarray), f"a_seq should be numpy array, got {type(a_seq)}"
        assert x_seq.dtype == np.int64, f"x_seq should be int64, got {x_seq.dtype}"
        assert a_seq.dtype == np.int64, f"a_seq should be int64, got {a_seq.dtype}"
        
        print("✅ NO CUDA CONVERSION ERRORS!")
        
        # Test with NumPy adapter for comparison
        print("\n📍 Testing RoomNPAdapter...")
        room_array = room_tensor.cpu().numpy()
        np_adapter = RoomNPAdapter(room_array, seed=42)
        
        start_time = time.time()
        x_np, a_np = np_adapter.generate_sequence(150000)
        elapsed = time.time() - start_time
        
        print(f"✅ NumPy adapter: Generated {len(x_np)} steps in {elapsed:.2f}s")
        
        # Test basic CHMM compatibility
        print("\n📍 Testing CHMM compatibility...")
        from models.chmm_torch import CHMM_torch
        from env_adapters.room_utils import get_room_n_clones
        
        # Use a smaller subset for quick CHMM test
        subset_size = 10000
        x_subset = torch.tensor(x_seq[:subset_size], dtype=torch.int64, device=device)
        a_subset = torch.tensor(a_seq[:subset_size], dtype=torch.int64, device=device)
        n_clones = get_room_n_clones(n_clones_per_obs=2, device=device)
        
        model = CHMM_torch(n_clones, x_subset, a_subset, seed=42)
        bps = model.bps(x_subset, a_subset)
        
        print(f"✅ CHMM model created successfully")
        print(f"✅ Initial BPS: {bps.item():.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ env.generate_sequence(150000) works perfectly")
        print("✅ No 'can't convert cuda:0 device type tensor to numpy' errors")
        print("✅ Proper tensor/numpy type safety maintained")
        print("✅ CHMM model integration working")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)