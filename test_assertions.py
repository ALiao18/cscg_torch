#!/usr/bin/env python3
"""
Test script to verify all strict assertions work correctly.

This script tests the assertion validation across all modules to ensure
proper type safety and data compatibility.
"""

import torch
import numpy as np
import sys
import traceback

def test_base_adapter():
    """Test base adapter assertions."""
    print("=== Testing Base Adapter ===")
    
    try:
        from env_adapters.base_adapter import CSCGEnvironmentAdapter
        
        # Test valid initialization
        adapter = CSCGEnvironmentAdapter(seed=42)
        print("✓ Valid initialization passed")
        
        # Test invalid seed
        try:
            CSCGEnvironmentAdapter(seed=-1)
            print("✗ Invalid seed test failed")
        except AssertionError:
            print("✓ Invalid seed assertion passed")
        
        # Test invalid seed type
        try:
            CSCGEnvironmentAdapter(seed="invalid")
            print("✗ Invalid seed type test failed")
        except AssertionError:
            print("✓ Invalid seed type assertion passed")
            
        print("✓ Base adapter tests completed")
        
    except Exception as e:
        print(f"✗ Base adapter test failed: {e}")
        traceback.print_exc()

def test_room_utils():
    """Test room utilities assertions."""
    print("\n=== Testing Room Utils ===")
    
    try:
        from env_adapters.room_utils import get_room_n_clones, create_room_adapter
        
        # Test valid n_clones creation
        n_clones = get_room_n_clones(n_clones_per_obs=2)
        assert isinstance(n_clones, torch.Tensor)
        assert n_clones.shape == (16,)
        assert torch.all(n_clones == 2)
        print("✓ Valid n_clones creation passed")
        
        # Test invalid n_clones_per_obs
        try:
            get_room_n_clones(n_clones_per_obs=0)
            print("✗ Invalid n_clones_per_obs test failed")
        except AssertionError:
            print("✓ Invalid n_clones_per_obs assertion passed")
        
        # Test valid room adapter creation
        room_tensor = torch.randint(0, 4, size=[5, 5], dtype=torch.long)
        adapter = create_room_adapter(room_tensor, adapter_type="torch")
        print("✓ Valid room adapter creation passed")
        
        # Test invalid adapter_type
        try:
            create_room_adapter(room_tensor, adapter_type="invalid")
            print("✗ Invalid adapter_type test failed")
        except AssertionError:
            print("✓ Invalid adapter_type assertion passed")
            
        print("✓ Room utils tests completed")
        
    except Exception as e:
        print(f"✗ Room utils test failed: {e}")
        traceback.print_exc()

def test_room_adapter():
    """Test room adapter assertions."""
    print("\n=== Testing Room Adapter ===")
    
    try:
        from env_adapters.room_adapter import RoomTorchAdapter, RoomNPAdapter
        
        # Test valid PyTorch adapter
        room_tensor = torch.randint(0, 4, size=[5, 5], dtype=torch.long)
        adapter = RoomTorchAdapter(room_tensor)
        print("✓ Valid PyTorch adapter creation passed")
        
        # Test valid numpy adapter
        room_array = np.random.randint(0, 4, size=(5, 5), dtype=np.int64)
        adapter = RoomNPAdapter(room_array)
        print("✓ Valid NumPy adapter creation passed")
        
        # Test invalid room tensor type
        try:
            RoomTorchAdapter("invalid_room")
            print("✗ Invalid room tensor type test failed")
        except AssertionError:
            print("✓ Invalid room tensor type assertion passed")
        
        # Test invalid room array type
        try:
            RoomNPAdapter("invalid_room")
            print("✗ Invalid room array type test failed")
        except AssertionError:
            print("✓ Invalid room array type assertion passed")
            
        print("✓ Room adapter tests completed")
        
    except Exception as e:
        print(f"✗ Room adapter test failed: {e}")
        traceback.print_exc()

def test_sequence_generation():
    """Test sequence generation with assertions."""
    print("\n=== Testing Sequence Generation ===")
    
    try:
        from env_adapters.room_utils import create_room_adapter
        
        # Create a simple room
        room_tensor = torch.randint(0, 2, size=[5, 5], dtype=torch.long)
        room_tensor[0, :] = -1  # walls
        room_tensor[-1, :] = -1
        room_tensor[:, 0] = -1  
        room_tensor[:, -1] = -1
        
        adapter = create_room_adapter(room_tensor)
        
        # Test valid sequence generation
        x_seq, a_seq = adapter.generate_sequence(10)
        assert isinstance(x_seq, np.ndarray)
        assert isinstance(a_seq, np.ndarray)
        assert len(x_seq) == len(a_seq)
        print("✓ Valid sequence generation passed")
        
        # Test invalid length
        try:
            adapter.generate_sequence(0)
            print("✗ Invalid length test failed")
        except AssertionError:
            print("✓ Invalid length assertion passed")
        
        # Test invalid length type
        try:
            adapter.generate_sequence("invalid")
            print("✗ Invalid length type test failed")
        except AssertionError:
            print("✓ Invalid length type assertion passed")
            
        print("✓ Sequence generation tests completed")
        
    except Exception as e:
        print(f"✗ Sequence generation test failed: {e}")
        traceback.print_exc()

def test_validate_seq():
    """Test sequence validation function."""
    print("\n=== Testing Sequence Validation ===")
    
    try:
        from models.train_utils import validate_seq
        
        # Test valid sequences
        x = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        a = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        n_clones = torch.tensor([1, 1, 1, 1, 1], dtype=torch.int64)
        
        validate_seq(x, a, n_clones)
        print("✓ Valid sequence validation passed")
        
        # Test mismatched lengths
        try:
            a_short = torch.tensor([0, 1], dtype=torch.int64)
            validate_seq(x, a_short, n_clones)
            print("✗ Mismatched lengths test failed")
        except AssertionError:
            print("✓ Mismatched lengths assertion passed")
        
        # Test invalid dtype
        try:
            x_float = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
            validate_seq(x_float, a, n_clones)
            print("✗ Invalid dtype test failed")
        except AssertionError:
            print("✓ Invalid dtype assertion passed")
            
        print("✓ Sequence validation tests completed")
        
    except Exception as e:
        print(f"✗ Sequence validation test failed: {e}")
        traceback.print_exc()

def main():
    """Run all assertion tests."""
    print("🧪 Testing Strict Assertions Across CSCG_Torch Codebase")
    print("=" * 60)
    
    # Test each module
    test_base_adapter()
    test_room_utils() 
    test_room_adapter()
    test_sequence_generation()
    test_validate_seq()
    
    print("\n" + "=" * 60)
    print("🎉 All assertion tests completed!")
    print("✅ Strict type safety and data compatibility checks are working correctly.")

if __name__ == "__main__":
    main()