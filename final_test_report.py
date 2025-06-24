#!/usr/bin/env python3
"""
Final Test Report for CSCG_Torch

This script provides a comprehensive verification that all systems work correctly,
particularly focusing on the env.generate_sequence(150000) test and CUDA safety.
"""

import torch
import numpy as np
import time
import traceback

def run_final_verification():
    """Run the final comprehensive verification."""
    print("🏁 FINAL VERIFICATION: CSCG_TORCH COMPREHENSIVE TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    results = []
    
    # Test 1: Import Verification
    print("\n🔍 Test 1: Import Verification")
    try:
        from env_adapters.room_adapter import RoomTorchAdapter, RoomNPAdapter
        from env_adapters.room_utils import create_room_adapter, get_room_n_clones
        from models.chmm_torch import CHMM_torch
        from models.train_utils import validate_seq
        results.append(("Import Verification", "✅ PASS"))
        print("   ✅ All modules import successfully")
    except Exception as e:
        results.append(("Import Verification", f"❌ FAIL: {e}"))
        print(f"   ❌ Import failed: {e}")
    
    # Test 2: Room Creation and Basic Functionality
    print("\n🏠 Test 2: Room Creation and Basic Functionality")
    try:
        room_tensor = torch.randint(0, 4, size=[20, 20], dtype=torch.long, device=device)
        room_tensor[0, :] = -1  # walls
        room_tensor[-1, :] = -1
        room_tensor[:, 0] = -1
        room_tensor[:, -1] = -1
        
        torch_adapter = RoomTorchAdapter(room_tensor, seed=42)
        np_adapter = RoomNPAdapter(room_tensor.cpu().numpy(), seed=42)
        
        # Test basic operations
        obs1 = torch_adapter.reset()
        obs2, valid = torch_adapter.step(0)
        
        assert isinstance(obs1, int) and isinstance(obs2, int)
        assert isinstance(valid, bool)
        assert 0 <= obs1 <= 15 and 0 <= obs2 <= 15
        
        results.append(("Room Creation & Basic Ops", "✅ PASS"))
        print("   ✅ Room adapters created and basic operations work")
    except Exception as e:
        results.append(("Room Creation & Basic Ops", f"❌ FAIL: {e}"))
        print(f"   ❌ Room creation failed: {e}")
    
    # Test 3: MAIN TEST - Large Sequence Generation
    print("\n🎯 Test 3: MAIN TEST - Large Sequence Generation (150k steps)")
    try:
        start_time = time.time()
        x_large, a_large = torch_adapter.generate_sequence(150000)
        elapsed = time.time() - start_time
        
        # Critical validations
        assert isinstance(x_large, np.ndarray), f"x_large must be numpy array, got {type(x_large)}"
        assert isinstance(a_large, np.ndarray), f"a_large must be numpy array, got {type(a_large)}"
        assert x_large.dtype == np.int64, f"x_large dtype must be int64, got {x_large.dtype}"
        assert a_large.dtype == np.int64, f"a_large dtype must be int64, got {a_large.dtype}"
        assert len(x_large) == len(a_large), "Sequence lengths must match"
        assert len(x_large) > 100000, f"Sequence too short: {len(x_large)}"
        
        # Validate ranges
        assert np.all(x_large >= 0) and np.all(x_large <= 15), f"Invalid x_large range: {x_large.min()}-{x_large.max()}"
        assert np.all(a_large >= 0) and np.all(a_large <= 3), f"Invalid a_large range: {a_large.min()}-{a_large.max()}"
        
        results.append(("Large Sequence Generation", "✅ PASS"))
        print(f"   ✅ Generated {len(x_large)} steps in {elapsed:.2f}s")
        print(f"   ✅ NO 'can't convert cuda:0 device type tensor to numpy' errors!")
        print(f"   ✅ Proper types: x={type(x_large)}, a={type(a_large)}")
        print(f"   ✅ Valid ranges: x=[{x_large.min()}, {x_large.max()}], a=[{a_large.min()}, {a_large.max()}]")
        
    except Exception as e:
        results.append(("Large Sequence Generation", f"❌ FAIL: {e}"))
        print(f"   ❌ Large sequence generation failed: {e}")
        traceback.print_exc()
        return results
    
    # Test 4: CHMM Model Integration
    print("\n🧠 Test 4: CHMM Model Integration")
    try:
        # Use a subset for faster testing
        subset_size = 50000
        x_subset = torch.tensor(x_large[:subset_size], dtype=torch.int64, device=device)
        a_subset = torch.tensor(a_large[:subset_size], dtype=torch.int64, device=device)
        n_clones = get_room_n_clones(n_clones_per_obs=2, device=device)
        
        model = CHMM_torch(n_clones, x_subset, a_subset, seed=42)
        bps = model.bps(x_subset, a_subset)
        
        assert isinstance(bps, torch.Tensor), f"BPS must be tensor, got {type(bps)}"
        assert bps.ndim == 0, f"BPS must be scalar, got {bps.ndim}D"
        assert bps.item() > 0, f"BPS must be positive, got {bps.item()}"
        
        results.append(("CHMM Model Integration", "✅ PASS"))
        print(f"   ✅ CHMM model created with {subset_size} steps")
        print(f"   ✅ Initial BPS: {bps.item():.4f}")
        print(f"   ✅ Model device: {model.device}")
        
    except Exception as e:
        results.append(("CHMM Model Integration", f"❌ FAIL: {e}"))
        print(f"   ❌ CHMM integration failed: {e}")
        traceback.print_exc()
    
    # Test 5: Error Handling Verification
    print("\n⚠️  Test 5: Error Handling Verification")
    try:
        error_count = 0
        
        # Test invalid room tensor
        try:
            RoomTorchAdapter("invalid")
            print("   ❌ Should have caught invalid room tensor")
        except AssertionError:
            error_count += 1
        
        # Test invalid sequence length
        try:
            torch_adapter.generate_sequence(-1)
            print("   ❌ Should have caught invalid sequence length")
        except AssertionError:
            error_count += 1
        
        # Test invalid room dimensions
        try:
            bad_room = torch.tensor([1, 2, 3], dtype=torch.long)
            RoomTorchAdapter(bad_room)
            print("   ❌ Should have caught invalid room dimensions")
        except AssertionError:
            error_count += 1
        
        if error_count == 3:
            results.append(("Error Handling", "✅ PASS"))
            print("   ✅ All 3 error cases caught correctly")
        else:
            results.append(("Error Handling", f"❌ FAIL: Only {error_count}/3 errors caught"))
            print(f"   ❌ Only {error_count}/3 error cases caught")
            
    except Exception as e:
        results.append(("Error Handling", f"❌ FAIL: {e}"))
        print(f"   ❌ Error handling test failed: {e}")
    
    # Test 6: Memory and Performance
    print("\n💾 Test 6: Memory and Performance")
    try:
        import gc
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   📊 GPU Memory Usage: {gpu_memory:.2f}GB")
        
        # Test sequence generation speed
        start_time = time.time()
        x_speed, a_speed = torch_adapter.generate_sequence(10000)
        speed_elapsed = time.time() - start_time
        steps_per_second = len(x_speed) / speed_elapsed
        
        print(f"   ⚡ Generation Speed: {steps_per_second:.0f} steps/second")
        
        if steps_per_second > 100:  # Should be much faster than this
            results.append(("Memory & Performance", "✅ PASS"))
            print("   ✅ Performance is acceptable")
        else:
            results.append(("Memory & Performance", "⚠️  SLOW"))
            print("   ⚠️  Performance may be slow")
            
    except Exception as e:
        results.append(("Memory & Performance", f"❌ FAIL: {e}"))
        print(f"   ❌ Memory/performance test failed: {e}")
    
    # Print Final Results
    print("\n" + "=" * 70)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        print(f"{result:15} {test_name}")
        if "✅ PASS" in result:
            passed += 1
    
    print("=" * 70)
    print(f"📊 SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ env.generate_sequence(150000) works perfectly")
        print("✅ No CUDA tensor conversion errors")
        print("✅ All strict assertions working correctly")
        print("✅ CHMM integration successful")
        print("✅ Error handling robust")
        print("✅ System ready for production use")
    else:
        print("⚠️  Some tests failed - review results above")
    
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = run_final_verification()
    exit(0 if success else 1)