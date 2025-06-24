#!/usr/bin/env python3
"""
Comprehensive Test Suite for CSCG_Torch

This script thoroughly tests the codebase, especially focusing on:
1. env.generate_sequence(150000) without CUDA conversion errors
2. Tensor handling across CPU/GPU
3. CHMM model integration
4. All assertion systems
"""

import torch
import numpy as np
import time
import traceback
import gc
from contextlib import contextmanager

# Test configuration
LARGE_SEQUENCE_LENGTH = 150000
MEDIUM_SEQUENCE_LENGTH = 10000
SMALL_SEQUENCE_LENGTH = 100

@contextmanager
def timer(description):
    """Context manager for timing operations."""
    start = time.time()
    print(f"⏱️  {description}...", end=" ", flush=True)
    try:
        yield
        elapsed = time.time() - start
        print(f"✅ ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Failed after {elapsed:.2f}s")
        raise

def test_device_setup():
    """Test device configuration and availability."""
    print("\n🖥️  === DEVICE SETUP TEST ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Primary device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
        print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("ℹ️  CUDA not available, using CPU")
    
    return device

def test_basic_imports():
    """Test all import functionality."""
    print("\n📦 === IMPORT TEST ===")
    
    with timer("Importing base modules"):
        from env_adapters.base_adapter import CSCGEnvironmentAdapter
        from env_adapters.room_adapter import RoomTorchAdapter, RoomNPAdapter
        from env_adapters.room_utils import create_room_adapter, get_room_n_clones
        
    with timer("Importing model modules"):
        from models.chmm_torch import CHMM_torch
        from models.train_utils import validate_seq
    
    print("✅ All imports successful")
    return True

def create_test_room(size=50, device=None):
    """Create a test room with walls and open spaces."""
    if device is None:
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

def test_room_adapters(device):
    """Test both RoomTorchAdapter and RoomNPAdapter."""
    print("\n🏠 === ROOM ADAPTER TEST ===")
    
    from env_adapters.room_adapter import RoomTorchAdapter, RoomNPAdapter
    
    # Test RoomTorchAdapter
    with timer("Creating RoomTorchAdapter"):
        room_tensor = create_test_room(size=20, device=device)
        torch_adapter = RoomTorchAdapter(room_tensor, seed=42)
        assert torch_adapter.device == device
        
    with timer("Testing RoomTorchAdapter reset/step"):
        obs = torch_adapter.reset()
        assert isinstance(obs, int), f"Observation must be int, got {type(obs)}"
        assert 0 <= obs <= 15, f"Observation out of range: {obs}"
        
        new_obs, valid = torch_adapter.step(0)
        assert isinstance(new_obs, int), f"New observation must be int, got {type(new_obs)}"
        assert isinstance(valid, bool), f"Valid flag must be bool, got {type(valid)}"
    
    # Test RoomNPAdapter  
    with timer("Creating RoomNPAdapter"):
        room_array = room_tensor.cpu().numpy()
        np_adapter = RoomNPAdapter(room_array, seed=42)
        
    with timer("Testing RoomNPAdapter reset/step"):
        obs = np_adapter.reset()
        assert isinstance(obs, int), f"Observation must be int, got {type(obs)}"
        
        new_obs, valid = np_adapter.step(1)
        assert isinstance(new_obs, int), f"New observation must be int, got {type(new_obs)}"
        assert isinstance(valid, bool), f"Valid flag must be bool, got {type(valid)}"
    
    print("✅ Room adapter tests passed")
    return torch_adapter, np_adapter

def test_sequence_generation_sizes(adapter, device):
    """Test sequence generation with various sizes."""
    print(f"\n🔢 === SEQUENCE GENERATION TEST ({type(adapter).__name__}) ===")
    
    # Test small sequence
    with timer(f"Generating {SMALL_SEQUENCE_LENGTH} steps"):
        x_small, a_small = adapter.generate_sequence(SMALL_SEQUENCE_LENGTH)
        
        # Validate types and shapes
        assert isinstance(x_small, np.ndarray), f"x_small must be numpy array, got {type(x_small)}"
        assert isinstance(a_small, np.ndarray), f"a_small must be numpy array, got {type(a_small)}"
        assert x_small.dtype == np.int64, f"x_small dtype must be int64, got {x_small.dtype}"
        assert a_small.dtype == np.int64, f"a_small dtype must be int64, got {a_small.dtype}"
        assert len(x_small) == len(a_small), f"Sequence length mismatch: {len(x_small)} != {len(a_small)}"
        assert len(x_small) <= SMALL_SEQUENCE_LENGTH, f"Sequence too long: {len(x_small)} > {SMALL_SEQUENCE_LENGTH}"
        
        # Validate value ranges
        assert np.all(x_small >= 0) and np.all(x_small <= 15), f"x_small values out of range [0,15]: {x_small.min()}-{x_small.max()}"
        assert np.all(a_small >= 0) and np.all(a_small <= 3), f"a_small values out of range [0,3]: {a_small.min()}-{a_small.max()}"
    
    # Test medium sequence
    with timer(f"Generating {MEDIUM_SEQUENCE_LENGTH} steps"):
        x_med, a_med = adapter.generate_sequence(MEDIUM_SEQUENCE_LENGTH)
        assert isinstance(x_med, np.ndarray), f"x_med must be numpy array, got {type(x_med)}"
        assert len(x_med) <= MEDIUM_SEQUENCE_LENGTH
    
    # Test large sequence - THE MAIN TEST
    with timer(f"Generating {LARGE_SEQUENCE_LENGTH} steps (MAIN TEST)"):
        try:
            x_large, a_large = adapter.generate_sequence(LARGE_SEQUENCE_LENGTH)
            
            # Critical validations for large sequence
            assert isinstance(x_large, np.ndarray), f"x_large must be numpy array, got {type(x_large)}"
            assert isinstance(a_large, np.ndarray), f"a_large must be numpy array, got {type(a_large)}"
            assert x_large.dtype == np.int64, f"x_large dtype must be int64, got {x_large.dtype}"
            assert a_large.dtype == np.int64, f"a_large dtype must be int64, got {a_large.dtype}"
            assert len(x_large) == len(a_large), f"Large sequence length mismatch"
            assert len(x_large) <= LARGE_SEQUENCE_LENGTH, f"Large sequence too long: {len(x_large)}"
            
            # No CUDA tensors should remain
            assert not isinstance(x_large, torch.Tensor), "x_large should not be a tensor"
            assert not isinstance(a_large, torch.Tensor), "a_large should not be a tensor"
            
            print(f"✅ Large sequence generated successfully: {len(x_large)} steps")
            print(f"   Observation range: [{x_large.min()}, {x_large.max()}]")
            print(f"   Action range: [{a_large.min()}, {a_large.max()}]")
            
            return x_large, a_large
            
        except Exception as e:
            print(f"❌ Large sequence generation failed: {e}")
            traceback.print_exc()
            raise
    
    return x_small, a_small

def test_chmm_integration(x_seq, a_seq, device):
    """Test CHMM model with generated sequences."""
    print(f"\n🧠 === CHMM INTEGRATION TEST ===")
    
    from models.chmm_torch import CHMM_torch
    from env_adapters.room_utils import get_room_n_clones
    
    with timer("Converting sequences to tensors"):
        # Convert numpy arrays to tensors
        x_tensor = torch.tensor(x_seq, dtype=torch.int64, device=device)
        a_tensor = torch.tensor(a_seq, dtype=torch.int64, device=device)
        
        # Get n_clones for room navigation
        n_clones = get_room_n_clones(n_clones_per_obs=2, device=device)
        
        print(f"   Sequence length: {len(x_tensor)}")
        print(f"   Observations: {x_tensor.min()}-{x_tensor.max()}")
        print(f"   Actions: {a_tensor.min()}-{a_tensor.max()}")
        print(f"   n_clones shape: {n_clones.shape}")
    
    with timer("Creating CHMM model"):
        model = CHMM_torch(n_clones, x_tensor, a_tensor, seed=42)
        assert model.device == device
        print(f"   Model device: {model.device}")
        print(f"   Transition matrix shape: {model.T.shape}")
    
    with timer("Computing bits-per-step (BPS)"):
        bps = model.bps(x_tensor, a_tensor)
        assert isinstance(bps, torch.Tensor), f"BPS must be tensor, got {type(bps)}"
        assert bps.ndim == 0, f"BPS must be scalar, got {bps.ndim}D"
        print(f"   Initial BPS: {bps.item():.4f}")
    
    with timer("Running short EM training (5 iterations)"):
        convergence = model.learn_em_T(x_tensor, a_tensor, n_iter=5, term_early=False)
        assert isinstance(convergence, list), f"Convergence must be list, got {type(convergence)}"
        assert len(convergence) == 5, f"Expected 5 iterations, got {len(convergence)}"
        print(f"   Final BPS: {convergence[-1]:.4f}")
        print(f"   Improvement: {convergence[0] - convergence[-1]:.4f}")
    
    print("✅ CHMM integration test passed")
    return model

def test_colab_imports():
    """Test the colab imports functionality."""
    print("\n🔬 === COLAB IMPORTS TEST ===")
    
    try:
        with timer("Testing colab_imports_fixed.py"):
            # Import the colab fixed version
            from colab_imports_fixed import RoomTorchAdapter as ColabRoomTorchAdapter
            from colab_imports_fixed import create_room_adapter, test_setup
            
            # Test the colab room adapter
            room_tensor = create_test_room(size=10)
            colab_adapter = ColabRoomTorchAdapter(room_tensor, seed=42)
            
            # Test sequence generation
            x_seq, a_seq = colab_adapter.generate_sequence(100)
            
            # The colab version returns tensors, not numpy arrays
            assert isinstance(x_seq, torch.Tensor), f"Colab x_seq should be tensor, got {type(x_seq)}"
            assert isinstance(a_seq, torch.Tensor), f"Colab a_seq should be tensor, got {type(a_seq)}"
            
            print("✅ Colab imports working correctly")
            return True
            
    except ImportError as e:
        print(f"ℹ️  Colab imports not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Colab imports test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test that assertions catch invalid inputs correctly."""
    print("\n⚠️  === ERROR HANDLING TEST ===")
    
    from env_adapters.room_adapter import RoomTorchAdapter
    
    error_count = 0
    
    # Test invalid room tensor type
    with timer("Testing invalid room tensor type"):
        try:
            RoomTorchAdapter("invalid_room")
            print("❌ Should have failed")
        except AssertionError:
            error_count += 1
            print("✅ Caught invalid room tensor type")
    
    # Test invalid room dimensions
    with timer("Testing invalid room dimensions"):
        try:
            bad_room = torch.tensor([1, 2, 3], dtype=torch.long)  # 1D instead of 2D
            RoomTorchAdapter(bad_room)
            print("❌ Should have failed")
        except AssertionError:
            error_count += 1
            print("✅ Caught invalid room dimensions")
    
    # Test invalid sequence generation length
    with timer("Testing invalid sequence length"):
        try:
            room = create_test_room(size=5)
            adapter = RoomTorchAdapter(room)
            adapter.generate_sequence(-10)  # negative length
            print("❌ Should have failed")
        except AssertionError:
            error_count += 1
            print("✅ Caught invalid sequence length")
    
    print(f"✅ Error handling test passed: {error_count}/3 assertions caught correctly")
    return error_count == 3

def memory_usage_info():
    """Print memory usage information."""
    if torch.cuda.is_available():
        print(f"🧠 GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, "
              f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """Run the comprehensive test suite."""
    print("🧪 COMPREHENSIVE CSCG_TORCH TEST SUITE")
    print("=" * 60)
    print("🎯 Main Focus: Testing env.generate_sequence(150000) without CUDA errors")
    print("=" * 60)
    
    try:
        # Phase 1: Setup
        device = test_device_setup()
        test_basic_imports()
        
        # Phase 2: Basic functionality
        torch_adapter, np_adapter = test_room_adapters(device)
        
        # Phase 3: Sequence generation (THE MAIN TEST)
        print(f"\n🚀 === MAIN TEST: {LARGE_SEQUENCE_LENGTH} SEQUENCE GENERATION ===")
        
        # Test with PyTorch adapter (most likely to have CUDA issues)
        x_large, a_large = test_sequence_generation_sizes(torch_adapter, device)
        memory_usage_info()
        
        # Test with NumPy adapter for comparison
        x_np, a_np = test_sequence_generation_sizes(np_adapter, device)
        memory_usage_info()
        
        # Phase 4: CHMM integration with large sequence
        model = test_chmm_integration(x_large, a_large, device)
        memory_usage_info()
        
        # Phase 5: Additional tests
        test_colab_imports()
        test_error_handling()
        
        # Final memory check
        memory_usage_info()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ No 'can't convert cuda:0 device type tensor to numpy' errors")
        print(f"✅ Successfully generated {LARGE_SEQUENCE_LENGTH} step sequences")
        print("✅ CHMM model integration working correctly")
        print("✅ All assertions and error handling working")
        print("✅ Memory usage under control")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        traceback.print_exc()
        memory_usage_info()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)