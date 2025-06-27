"""
Comprehensive Tests for Room Environment Adapters

Tests for room-based environment adapters including:
- Base adapter functionality
- NumPy and PyTorch room adapters
- Room utilities and helper functions
- Integration with test room data
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cscg_torch.env_adapters.base_adapter import CSCGEnvironmentAdapter
from cscg_torch.env_adapters.room_adapter import RoomNPAdapter, RoomTorchAdapter
from cscg_torch.env_adapters.room_utils import (
    create_room_adapter, get_room_n_clones, demo_room_setup
)
from tests.test_config import (
    TestConfig, setup_test_environment, cleanup_test_environment,
    load_test_room, check_tensor_properties, measure_performance,
    check_gpu_memory, gpu_test, slow_test
)

class TestBaseAdapter:
    """Test base environment adapter functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        setup_test_environment()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_base_adapter_interface(self):
        """Test that base adapter defines the correct interface."""
        try:
            # Create base adapter instance
            adapter = CSCGEnvironmentAdapter(seed=TestConfig.SEED)
            
            # Check required attributes
            assert hasattr(adapter, 'device'), "Base adapter missing device attribute"
            assert hasattr(adapter, 'rng'), "Base adapter missing rng attribute"
            assert hasattr(adapter, 'n_actions'), "Base adapter missing n_actions attribute"
            
            # Check required methods exist (even if not implemented)
            assert hasattr(adapter, 'reset'), "Base adapter missing reset method"
            assert hasattr(adapter, 'step'), "Base adapter missing step method"
            assert hasattr(adapter, 'get_observation'), "Base adapter missing get_observation method"
            assert hasattr(adapter, 'generate_sequence'), "Base adapter missing generate_sequence method"
            
            print("✓ Base adapter interface test passed")
            
        except Exception as e:
            print(f"✗ Base adapter interface test failed: {e}")
            raise
    
    def test_base_adapter_initialization(self):
        """Test base adapter initialization."""
        try:
            # Test with different seeds
            adapter1 = CSCGEnvironmentAdapter(seed=42)
            adapter2 = CSCGEnvironmentAdapter(seed=123)
            
            # Check that different seeds create different random states
            assert adapter1.rng.get_state()[1][0] != adapter2.rng.get_state()[1][0], \
                "Different seeds should create different random states"
            
            # Check device setting
            assert isinstance(adapter1.device, torch.device), "Device should be torch.device"
            
            print("✓ Base adapter initialization test passed")
            
        except Exception as e:
            print(f"✗ Base adapter initialization test failed: {e}")
            raise

class TestRoomAdapters:
    """Test room-specific adapters."""
    
    def setup_method(self):
        """Set up test environment and load test room."""
        setup_test_environment()
        self.test_room_tensor = load_test_room(TestConfig.ROOM_SIZE)
        self.test_room_numpy = self.test_room_tensor.numpy()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_room_np_adapter_basic(self):
        """Test basic NumPy room adapter functionality."""
        try:
            # Create NumPy adapter
            adapter = RoomNPAdapter(self.test_room_numpy, seed=TestConfig.SEED)
            
            # Check basic properties
            assert adapter.n_actions == 4, f"Expected 4 actions, got {adapter.n_actions}"
            assert adapter.h == TestConfig.ROOM_SIZE, f"Height mismatch: {adapter.h} != {TestConfig.ROOM_SIZE}"
            assert adapter.w == TestConfig.ROOM_SIZE, f"Width mismatch: {adapter.w} != {TestConfig.ROOM_SIZE}"
            
            # Test reset
            obs = adapter.reset()
            assert isinstance(obs, int), f"Observation must be int, got {type(obs)}"
            assert 0 <= obs <= 15, f"Observation out of range: {obs}"
            
            print(f"✓ NumPy room adapter basic test passed")
            print(f"  Room size: {adapter.h}x{adapter.w}")
            print(f"  Initial observation: {obs}")
            
        except Exception as e:
            print(f"✗ NumPy room adapter basic test failed: {e}")
            raise
    
    def test_room_torch_adapter_basic(self):
        """Test basic PyTorch room adapter functionality."""
        try:
            # Create PyTorch adapter
            adapter = RoomTorchAdapter(self.test_room_tensor, seed=TestConfig.SEED)
            
            # Check basic properties
            assert adapter.n_actions == 4, f"Expected 4 actions, got {adapter.n_actions}"
            assert adapter.h == TestConfig.ROOM_SIZE, f"Height mismatch: {adapter.h} != {TestConfig.ROOM_SIZE}"
            assert adapter.w == TestConfig.ROOM_SIZE, f"Width mismatch: {adapter.w} != {TestConfig.ROOM_SIZE}"
            
            # Check device placement
            assert adapter.room.device == adapter.device, "Room tensor device mismatch"
            
            # Test reset
            obs = adapter.reset()
            assert isinstance(obs, int), f"Observation must be int, got {type(obs)}"
            assert 0 <= obs <= 15, f"Observation out of range: {obs}"
            
            print(f"✓ PyTorch room adapter basic test passed")
            print(f"  Room size: {adapter.h}x{adapter.w}")
            print(f"  Device: {adapter.device}")
            print(f"  Initial observation: {obs}")
            
        except Exception as e:
            print(f"✗ PyTorch room adapter basic test failed: {e}")
            raise
    
    def test_room_adapter_movement(self):
        """Test movement mechanics in room adapters."""
        try:
            adapter = RoomNPAdapter(self.test_room_numpy, seed=TestConfig.SEED)
            
            # Reset to known position
            initial_obs = adapter.reset()
            initial_pos = adapter.pos
            
            # Test all actions
            action_names = ["up", "down", "left", "right"]
            movement_results = []
            
            for action in range(4):
                # Reset to initial position
                adapter.pos = initial_pos
                
                # Try action
                new_obs, valid = adapter.step(action)
                
                movement_results.append({
                    'action': action,
                    'action_name': action_names[action],
                    'valid': valid,
                    'new_obs': new_obs,
                    'new_pos': adapter.pos
                })
                
                # Validate observation
                assert isinstance(new_obs, int), f"Observation must be int for action {action}"
                assert 0 <= new_obs <= 15, f"Observation out of range for action {action}: {new_obs}"
            
            print(f"✓ Room adapter movement test passed")
            print(f"  Initial position: {initial_pos}")
            for result in movement_results:
                print(f"  {result['action_name']}: valid={result['valid']}, pos={result['new_pos']}")
            
        except Exception as e:
            print(f"✗ Room adapter movement test failed: {e}")
            raise
    
    def test_sequence_generation(self):
        """Test sequence generation functionality."""
        try:
            adapter = RoomNPAdapter(self.test_room_numpy, seed=TestConfig.SEED)
            
            # Generate test sequence
            sequence_length = 1000
            x_seq, a_seq, generation_time = measure_performance(
                adapter.generate_sequence, sequence_length
            )
            
            # Validate sequence properties
            assert isinstance(x_seq, np.ndarray), "x_seq must be numpy array"
            assert isinstance(a_seq, np.ndarray), "a_seq must be numpy array"
            assert len(x_seq) == len(a_seq), f"Sequence length mismatch: {len(x_seq)} != {len(a_seq)}"
            assert len(x_seq) <= sequence_length, f"Sequence too long: {len(x_seq)} > {sequence_length}"
            assert len(x_seq) > 0, "Sequence cannot be empty"
            
            # Validate observation and action ranges
            assert np.all(x_seq >= 0) and np.all(x_seq <= 15), "Observations out of range [0, 15]"
            assert np.all(a_seq >= 0) and np.all(a_seq <= 3), "Actions out of range [0, 3]"
            
            # Check data types
            assert x_seq.dtype == np.int64, f"x_seq dtype mismatch: {x_seq.dtype} != int64"
            assert a_seq.dtype == np.int64, f"a_seq dtype mismatch: {a_seq.dtype} != int64"
            
            print(f"✓ Sequence generation test passed")
            print(f"  Requested length: {sequence_length}")
            print(f"  Generated length: {len(x_seq)}")
            print(f"  Generation time: {generation_time:.3f}s")
            print(f"  Obs range: {x_seq.min()} to {x_seq.max()}")
            print(f"  Action range: {a_seq.min()} to {a_seq.max()}")
            
        except Exception as e:
            print(f"✗ Sequence generation test failed: {e}")
            raise
    
    @gpu_test
    def test_gpu_torch_adapter(self):
        """Test PyTorch adapter GPU functionality."""
        try:
            # Create GPU adapter
            adapter = RoomTorchAdapter(self.test_room_tensor, seed=TestConfig.SEED)
            
            # Check GPU placement
            if TestConfig.USE_GPU:
                assert adapter.device.type == 'cuda', "Adapter should use GPU when available"
                assert adapter.room.device.type == 'cuda', "Room tensor should be on GPU"
            
            # Test basic functionality on GPU
            obs = adapter.reset()
            new_obs, valid = adapter.step(0)  # Try moving up
            
            # Generate short sequence
            x_seq, a_seq = adapter.generate_sequence(100)
            
            print(f"✓ GPU PyTorch adapter test passed")
            print(f"  Device: {adapter.device}")
            print(f"  Generated sequence length: {len(x_seq)}")
            
        except Exception as e:
            print(f"✗ GPU PyTorch adapter test failed: {e}")
            raise

class TestRoomUtils:
    """Test room utility functions."""
    
    def setup_method(self):
        """Set up test environment."""
        setup_test_environment()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_create_room_adapter(self):
        """Test room adapter creation utility."""
        try:
            # Load test room data
            room_tensor = load_test_room()
            room_numpy = room_tensor.numpy()
            
            # Test PyTorch adapter creation
            torch_adapter = create_room_adapter(room_tensor, adapter_type="torch")
            assert isinstance(torch_adapter, RoomTorchAdapter), "Should create RoomTorchAdapter"
            
            # Test NumPy adapter creation
            numpy_adapter = create_room_adapter(room_numpy, adapter_type="numpy")
            assert isinstance(numpy_adapter, RoomNPAdapter), "Should create RoomNPAdapter"
            
            # Test automatic conversion
            torch_from_numpy = create_room_adapter(room_numpy, adapter_type="torch")
            assert isinstance(torch_from_numpy, RoomTorchAdapter), "Should convert numpy to torch adapter"
            
            numpy_from_torch = create_room_adapter(room_tensor, adapter_type="numpy")
            assert isinstance(numpy_from_torch, RoomNPAdapter), "Should convert torch to numpy adapter"
            
            print("✓ Room adapter creation test passed")
            
        except Exception as e:
            print(f"✗ Room adapter creation test failed: {e}")
            raise
    
    def test_get_room_n_clones(self):
        """Test n_clones generation for room environments."""
        try:
            # Test default parameters
            n_clones = get_room_n_clones()
            
            # Validate properties
            assert isinstance(n_clones, torch.Tensor), "n_clones must be tensor"
            assert n_clones.shape == (16,), f"n_clones shape mismatch: {n_clones.shape}"
            assert n_clones.dtype == torch.int64, f"n_clones dtype mismatch: {n_clones.dtype}"
            assert torch.all(n_clones > 0), "All n_clones values must be positive"
            
            # Test custom parameters
            custom_clones = get_room_n_clones(n_clones_per_obs=5)
            assert torch.all(custom_clones == 5), "Custom n_clones values incorrect"
            
            # Test device placement
            if TestConfig.USE_GPU:
                gpu_clones = get_room_n_clones(device=torch.device('cuda'))
                assert gpu_clones.device.type == 'cuda', "n_clones should be on specified device"
            
            print("✓ n_clones generation test passed")
            print(f"  Default n_clones: {n_clones[0].item()} per observation")
            print(f"  Total states: {n_clones.sum().item()}")
            
        except Exception as e:
            print(f"✗ n_clones generation test failed: {e}")
            raise
    
    def test_demo_room_setup(self):
        """Test demo room setup utility."""
        try:
            # Run demo setup
            adapter, n_clones, sample_data, setup_time = measure_performance(demo_room_setup)
            
            # Validate adapter
            assert hasattr(adapter, 'reset'), "Demo adapter missing reset method"
            assert hasattr(adapter, 'generate_sequence'), "Demo adapter missing generate_sequence method"
            
            # Validate n_clones
            assert isinstance(n_clones, torch.Tensor), "n_clones must be tensor"
            assert n_clones.shape == (16,), "n_clones shape incorrect"
            
            # Validate sample data
            x_seq, a_seq = sample_data
            assert isinstance(x_seq, np.ndarray), "x_seq must be numpy array"
            assert isinstance(a_seq, np.ndarray), "a_seq must be numpy array"
            assert len(x_seq) == len(a_seq), "Sample sequence length mismatch"
            assert len(x_seq) > 0, "Sample sequence cannot be empty"
            
            print("✓ Demo room setup test passed")
            print(f"  Setup time: {setup_time:.3f}s")
            print(f"  Sample sequence length: {len(x_seq)}")
            print(f"  Adapter type: {type(adapter).__name__}")
            
        except Exception as e:
            print(f"✗ Demo room setup test failed: {e}")
            raise

class TestRoomAdapterIntegration:
    """Test integration between room adapters and other components."""
    
    def setup_method(self):
        """Set up test environment."""
        setup_test_environment()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_adapter_with_different_room_sizes(self):
        """Test adapters with different room sizes."""
        try:
            test_sizes = [5, 10, 20]  # Skip 50x50 for speed
            
            for size in test_sizes:
                print(f"\n  Testing {size}x{size} room...")
                
                # Load room of specific size
                room = load_test_room(size)
                
                # Create adapter
                adapter = create_room_adapter(room, adapter_type="torch")
                
                # Test basic functionality
                obs = adapter.reset()
                assert 0 <= obs <= 15, f"Invalid observation for {size}x{size} room: {obs}"
                
                # Generate short sequence
                x_seq, a_seq = adapter.generate_sequence(100)
                assert len(x_seq) > 0, f"Empty sequence for {size}x{size} room"
                
                print(f"    ✓ {size}x{size} room test passed (obs: {obs}, seq_len: {len(x_seq)})")
            
            print("✓ Multi-size room adapter test passed")
            
        except Exception as e:
            print(f"✗ Multi-size room adapter test failed: {e}")
            raise
    
    @slow_test
    def test_large_sequence_generation(self):
        """Test generation of large sequences for performance."""
        try:
            # Use 5x5 room for fastest iteration
            room = load_test_room(5)
            adapter = create_room_adapter(room, adapter_type="torch")
            
            # Generate large sequence
            large_length = 10000
            x_seq, a_seq, generation_time = measure_performance(
                adapter.generate_sequence, large_length
            )
            
            # Validate large sequence
            assert len(x_seq) > large_length * 0.8, "Large sequence too short (possible early termination)"
            assert np.all(x_seq >= 0) and np.all(x_seq <= 15), "Large sequence observations out of range"
            assert np.all(a_seq >= 0) and np.all(a_seq <= 3), "Large sequence actions out of range"
            
            # Performance check
            generation_rate = len(x_seq) / generation_time
            assert generation_rate > 1000, f"Generation too slow: {generation_rate:.1f} steps/sec"
            
            print(f"✓ Large sequence generation test passed")
            print(f"  Sequence length: {len(x_seq)}")
            print(f"  Generation time: {generation_time:.2f}s")
            print(f"  Generation rate: {generation_rate:.1f} steps/sec")
            
        except Exception as e:
            print(f"✗ Large sequence generation test failed: {e}")
            raise

def test_plot_graph_functions():
    """Test plotting functions from base_adapter."""
    pytest.importorskip("matplotlib")
    
    try:
        from cscg_torch.env_adapters.base_adapter import plot_graph
        from cscg_torch.models.chmm_torch import CHMM_torch
        import tempfile
        import os
        
        # Create test data
        device = torch.device("cpu")
        x, a = create_test_sequence(100)
        n_clones = create_test_n_clones()
        
        # Create a simple CHMM for testing
        model = CHMM_torch(n_clones, x, a, device=device)
        
        # Test room plot
        room = torch.randint(0, 4, (8, 8), device=device)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test room mode
            plot_graph(
                model, room=room, plot_mode='room',
                trial_name='test_room', k=5
            )
            
            # Test progression mode
            progression = [10.5, 9.8, 9.2, 8.9, 8.7]
            plot_graph(
                model, progression=progression, plot_mode='progression',
                trial_name='test_progression', k=5
            )
            
            # Test usage mode
            plot_graph(
                model, x=x, a=a, plot_mode='usage',
                trial_name='test_usage', k=5
            )
            
            # Test performance mode
            plot_graph(
                model, x=x, a=a, plot_mode='performance',
                trial_name='test_performance', k=5
            )
            
        print("Plot graph functions test passed")
        
    except ImportError as e:
        pytest.skip(f"Required plotting dependencies not available: {e}")


def test_save_room_plot():
    """Test save_room_plot function."""
    pytest.importorskip("matplotlib")
    
    try:
        from cscg_torch.env_adapters.room_adapter import save_room_plot
        import tempfile
        import os
        
        # Create test room
        room = torch.randint(0, 3, (6, 6))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_room")
            save_room_plot(room, save_path, cmap='viridis')
            
            # Check that files were created
            assert os.path.exists(f"{save_path}.pdf"), "PDF file not created"
            assert os.path.exists(f"{save_path}.png"), "PNG file not created"
            
        print("Save room plot test passed")
        
    except ImportError as e:
        pytest.skip(f"Required plotting dependencies not available: {e}")


def test_room_utility_functions():
    """Test room utility functions."""
    try:
        from cscg_torch.env_adapters.room_utils import (
            get_obs_colormap, clone_to_obs_map, top_k_used_clones, count_used_clones
        )
        
        # Test get_obs_colormap
        cmap = get_obs_colormap(16)
        assert cmap is not None, "Colormap should not be None"
        
        # Test clone_to_obs_map
        n_clones = torch.tensor([3, 2, 4])
        mapping = clone_to_obs_map(n_clones)
        assert len(mapping) == 9, f"Mapping length mismatch: {len(mapping)} != 9"
        assert mapping[0] == 0 and mapping[1] == 0 and mapping[2] == 0, "First obs mapping wrong"
        assert mapping[3] == 1 and mapping[4] == 1, "Second obs mapping wrong"
        assert mapping[5] == 2 and mapping[8] == 2, "Third obs mapping wrong"
        
        # Test top_k_used_clones
        states = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
        top_clones = top_k_used_clones(states, k=3)
        assert len(top_clones) <= 3, f"Too many top clones: {len(top_clones)}"
        assert top_clones[0][0] == 3, "Most frequent clone should be 3"
        assert top_clones[0][1] == 4, "Clone 3 should appear 4 times"
        
        print("Room utility functions test passed")
        
    except ImportError as e:
        pytest.skip(f"Required utility dependencies not available: {e}")


def test_room_utils_with_chmm_integration():
    """Test room utilities integration with CHMM model."""
    try:
        from cscg_torch.env_adapters.room_utils import count_used_clones
        from cscg_torch.models.chmm_torch import CHMM_torch
        
        # Create test data
        device = torch.device("cpu")
        x, a = create_test_sequence(50)
        n_clones = create_test_n_clones()
        
        # Create model
        model = CHMM_torch(n_clones, x, a, device=device)
        
        # Test count_used_clones
        usage_counts = count_used_clones(model, x, a)
        
        assert isinstance(usage_counts, dict), "Usage counts should be a dict"
        assert len(usage_counts) == len(n_clones), f"Usage counts length mismatch: {len(usage_counts)} != {len(n_clones)}"
        assert all(isinstance(k, int) and isinstance(v, int) for k, v in usage_counts.items()), "Usage counts should have int keys and values"
        assert all(v >= 0 for v in usage_counts.values()), "All usage counts should be non-negative"
        
        print("Room utils CHMM integration test passed")
        
    except ImportError as e:
        pytest.skip(f"Required integration dependencies not available: {e}")


def test_plotting_error_handling():
    """Test error handling in plotting functions."""
    try:
        from cscg_torch.env_adapters.base_adapter import plot_graph
        from cscg_torch.models.chmm_torch import CHMM_torch
        
        device = torch.device("cpu")
        x, a = create_test_sequence(50)
        n_clones = create_test_n_clones()
        model = CHMM_torch(n_clones, x, a, device=device)
        
        # Test invalid plot mode
        with pytest.raises(ValueError, match="Unsupported plot_mode"):
            plot_graph(model, plot_mode='invalid_mode')
        
        # Test missing required data
        with pytest.raises(AssertionError, match="room required for room plot"):
            plot_graph(model, plot_mode='room')
            
        with pytest.raises(AssertionError, match="x and a required"):
            plot_graph(model, plot_mode='usage')
        
        print("Plotting error handling test passed")
        
    except ImportError as e:
        pytest.skip(f"Required plotting dependencies not available: {e}")

def run_all_tests():
    """Run all room adapter tests."""
    print("Room Environment Adapters Test Suite")
    print("=" * 50)
    
    test_classes = [
        TestBaseAdapter(),
        TestRoomAdapters(),
        TestRoomUtils(),
        TestRoomAdapterIntegration()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}")
        print("-" * len(class_name))
        
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Set up test
                if hasattr(test_class, 'setup_method'):
                    test_class.setup_method()
                
                # Run test
                method = getattr(test_class, method_name)
                method()
                passed_tests += 1
                
                # Tear down test
                if hasattr(test_class, 'teardown_method'):
                    test_class.teardown_method()
                    
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                # Continue with other tests
                if hasattr(test_class, 'teardown_method'):
                    test_class.teardown_method()
    
    print("\n\nTest Summary")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

if __name__ == "__main__":
    run_all_tests()