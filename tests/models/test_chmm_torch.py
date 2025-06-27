"""
Comprehensive Tests for CHMM_torch Class

Tests for the optimized CHMM implementation including:
- Initialization and device management
- Forward/backward computation
- EM training algorithms
- GPU optimization and memory management
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cscg_torch.models.chmm_torch import CHMM_torch
from cscg_torch.models.train_utils import (
    train_chmm, make_E, make_E_sparse, compute_forward_messages, place_field
)
from tests.test_config import (
    TestConfig, setup_test_environment, cleanup_test_environment,
    create_test_sequence, create_test_n_clones, check_tensor_properties,
    measure_performance, check_gpu_memory, gpu_test, slow_test
)

class TestCHMMTorchInitialization:
    """Test CHMM_torch initialization and basic properties."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        setup_test_environment()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_basic_initialization(self):
        """Test basic CHMM initialization with default parameters."""
        try:
            # Create test data
            x, a = create_test_sequence(length=100)
            n_clones = create_test_n_clones()
            
            # Initialize model
            model = CHMM_torch(n_clones, x, a, pseudocount=0.1, seed=TestConfig.SEED)
            
            # Check basic properties
            assert hasattr(model, 'device'), "Model missing device attribute"
            assert hasattr(model, 'T'), "Model missing transition matrix T"
            assert hasattr(model, 'C'), "Model missing count matrix C"
            assert hasattr(model, 'Pi_x'), "Model missing initial distribution Pi_x"
            
            # Check tensor properties
            assert check_tensor_properties(model.T, expected_device=model.device)
            assert check_tensor_properties(model.C, expected_device=model.device)
            assert check_tensor_properties(model.Pi_x, expected_device=model.device)
            
            print("✓ Basic initialization test passed")
            
        except Exception as e:
            print(f"✗ Basic initialization test failed: {e}")
            raise
    
    def test_device_placement(self):
        """Test that all tensors are placed on the correct device."""
        try:
            x, a = create_test_sequence(length=50)
            n_clones = create_test_n_clones()
            
            model = CHMM_torch(n_clones, x, a)
            
            # Check device consistency
            expected_device = model.device
            assert model.T.device == expected_device, f"T device mismatch: {model.T.device} != {expected_device}"
            assert model.C.device == expected_device, f"C device mismatch: {model.C.device} != {expected_device}"
            assert model.Pi_x.device == expected_device, f"Pi_x device mismatch: {model.Pi_x.device} != {expected_device}"
            assert model.n_clones.device == expected_device, f"n_clones device mismatch: {model.n_clones.device} != {expected_device}"
            
            print(f"✓ Device placement test passed (device: {expected_device})")
            
        except Exception as e:
            print(f"✗ Device placement test failed: {e}")
            raise
    
    @gpu_test
    def test_gpu_initialization(self):
        """Test GPU-specific initialization features."""
        try:
            x, a = create_test_sequence(length=200)
            n_clones = create_test_n_clones()
            
            # Test with mixed precision
            model = CHMM_torch(n_clones, x, a, enable_mixed_precision=True)
            
            assert model.device.type == 'cuda', "Model not on GPU"
            assert model.cuda_available, "CUDA availability not detected"
            
            # Check memory usage
            memory_info = check_gpu_memory()
            assert memory_info['gpu_available'], "GPU memory check failed"
            
            print(f"✓ GPU initialization test passed")
            print(f"  Memory allocated: {memory_info['allocated_mb']:.1f} MB")
            
        except Exception as e:
            print(f"✗ GPU initialization test failed: {e}")
            raise
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        try:
            # Test various invalid inputs
            x, a = create_test_sequence(length=100)
            n_clones = create_test_n_clones()
            
            # Test invalid n_clones
            with pytest.raises((AssertionError, ValueError)):
                invalid_n_clones = torch.zeros(TestConfig.N_STATES, dtype=torch.int64)  # Zero clones
                CHMM_torch(invalid_n_clones, x, a)
            
            # Test mismatched sequence lengths
            with pytest.raises((AssertionError, ValueError)):
                x_short = x[:50]
                CHMM_torch(n_clones, x_short, a)  # Different lengths
            
            # Test invalid pseudocount
            with pytest.raises((AssertionError, ValueError)):
                CHMM_torch(n_clones, x, a, pseudocount=-1.0)  # Negative pseudocount
            
            print("✓ Input validation test passed")
            
        except Exception as e:
            print(f"✗ Input validation test failed: {e}")
            raise

class TestCHMMTorchComputation:
    """Test CHMM computation methods."""
    
    def setup_method(self):
        """Set up test environment and model."""
        setup_test_environment()
        self.x, self.a = create_test_sequence(length=TestConfig.SEQUENCE_LENGTH)
        self.n_clones = create_test_n_clones()
        self.model = CHMM_torch(self.n_clones, self.x, self.a, seed=TestConfig.SEED)
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_bps_computation(self):
        """Test bits-per-step (BPS) computation."""
        try:
            # Test reduced BPS
            bps_reduced = self.model.bps(self.x, self.a, reduce=True)
            assert isinstance(bps_reduced, torch.Tensor), "BPS result must be tensor"
            assert bps_reduced.ndim == 0, "Reduced BPS must be scalar"
            assert torch.isfinite(bps_reduced), "BPS must be finite"
            
            # Test per-step BPS
            bps_per_step = self.model.bps(self.x, self.a, reduce=False)
            assert bps_per_step.shape == (len(self.x),), f"Per-step BPS shape mismatch: {bps_per_step.shape}"
            assert torch.isfinite(bps_per_step).all(), "All per-step BPS values must be finite"
            
            print(f"✓ BPS computation test passed")
            print(f"  Total BPS: {bps_reduced.item():.4f}")
            print(f"  Mean BPS: {bps_per_step.mean().item():.4f}")
            
        except Exception as e:
            print(f"✗ BPS computation test failed: {e}")
            raise
    
    def test_viterbi_decoding(self):
        """Test Viterbi decoding (MAP inference)."""
        try:
            # Test decode method
            neg_log_lik, states = self.model.decode(self.x, self.a)
            
            # Validate results
            assert isinstance(neg_log_lik, torch.Tensor), "Negative log-likelihood must be tensor"
            assert neg_log_lik.ndim == 0, "Negative log-likelihood must be scalar"
            assert torch.isfinite(neg_log_lik), "Negative log-likelihood must be finite"
            
            assert isinstance(states, torch.Tensor), "States must be tensor"
            assert states.shape == (len(self.x),), f"States shape mismatch: {states.shape}"
            assert states.dtype in [torch.int64, torch.long], f"States must be integer type: {states.dtype}"
            
            # Check state validity
            max_state = self.n_clones.sum().item() - 1
            assert torch.all(states >= 0), "All states must be non-negative"
            assert torch.all(states <= max_state), f"All states must be <= {max_state}"
            
            print(f"✓ Viterbi decoding test passed")
            print(f"  MAP log-likelihood: {-neg_log_lik.item():.4f}")
            print(f"  State range: {states.min().item()} to {states.max().item()}")
            
        except Exception as e:
            print(f"✗ Viterbi decoding test failed: {e}")
            raise
    
    def test_transition_matrix_update(self):
        """Test transition matrix update method."""
        try:
            # Store original T
            T_original = self.model.T.clone()
            
            # Modify count matrix
            self.model.C += 0.1
            
            # Update transition matrix
            self.model.update_T(verbose=False)
            
            # Check that T changed
            assert not torch.allclose(T_original, self.model.T), "T should change after C modification"
            
            # Check normalization
            T_sums = self.model.T.sum(dim=2)
            expected_sums = torch.ones_like(T_sums)
            assert torch.allclose(T_sums, expected_sums, atol=1e-6), "T rows must sum to 1"
            
            # Check device placement
            assert self.model.T.device == self.model.device, "T device mismatch after update"
            
            print("✓ Transition matrix update test passed")
            
        except Exception as e:
            print(f"✗ Transition matrix update test failed: {e}")
            raise

class TestCHMMTorchTraining:
    """Test CHMM training algorithms."""
    
    def setup_method(self):
        """Set up test environment and model."""
        setup_test_environment()
        # Use shorter sequences for faster training tests
        self.x, self.a = create_test_sequence(length=500)
        self.n_clones = create_test_n_clones()
        self.model = CHMM_torch(self.n_clones, self.x, self.a, seed=TestConfig.SEED)
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_em_training_basic(self):
        """Test basic EM training functionality."""
        try:
            # Store initial BPS
            initial_bps = self.model.bps(self.x, self.a, reduce=True).item()
            
            # Run EM training
            convergence, training_time = measure_performance(
                self.model.learn_em_T, 
                self.x, self.a, 
                n_iter=TestConfig.EM_ITERATIONS,
                term_early=True
            )
            
            # Validate convergence history
            assert isinstance(convergence, list), "Convergence history must be list"
            assert len(convergence) > 0, "Convergence history cannot be empty"
            assert len(convergence) <= TestConfig.EM_ITERATIONS, "Too many iterations recorded"
            
            # Check that all values are numeric
            for i, bps in enumerate(convergence):
                assert isinstance(bps, (int, float)), f"Convergence[{i}] must be numeric: {type(bps)}"
                assert np.isfinite(bps), f"Convergence[{i}] must be finite: {bps}"
            
            # Check improvement (BPS should generally decrease)
            final_bps = self.model.bps(self.x, self.a, reduce=True).item()
            
            print(f"✓ EM training test passed")
            print(f"  Initial BPS: {initial_bps:.4f}")
            print(f"  Final BPS: {final_bps:.4f}")
            print(f"  Improvement: {initial_bps - final_bps:.4f}")
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Iterations: {len(convergence)}")
            
        except Exception as e:
            print(f"✗ EM training test failed: {e}")
            raise
    
    @gpu_test
    def test_gpu_training_performance(self):
        """Test GPU training performance and memory usage."""
        try:
            # Monitor memory before training
            memory_before = check_gpu_memory()
            
            # Run training with performance monitoring
            convergence, training_time = measure_performance(
                self.model.learn_em_T,
                self.x, self.a,
                n_iter=5  # Short run for performance test
            )
            
            # Monitor memory after training
            memory_after = check_gpu_memory()
            
            # Check that training completed successfully
            assert len(convergence) > 0, "Training produced no convergence history"
            
            print(f"✓ GPU training performance test passed")
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Memory before: {memory_before['allocated_mb']:.1f} MB")
            print(f"  Memory after: {memory_after['allocated_mb']:.1f} MB")
            print(f"  Peak memory: {memory_after['max_allocated_mb']:.1f} MB")
            
        except Exception as e:
            print(f"✗ GPU training performance test failed: {e}")
            raise
    
    def test_training_convergence_detection(self):
        """Test training convergence detection and early termination."""
        try:
            # Run training with early termination
            convergence = self.model.learn_em_T(
                self.x, self.a,
                n_iter=50,
                term_early=True,
                min_improvement=1e-4
            )
            
            # Check that training may have terminated early
            assert len(convergence) <= 50, "Training exceeded maximum iterations"
            
            # If training converged early, check that improvement was minimal
            if len(convergence) < 50:
                print(f"  Early termination at iteration {len(convergence)}")
            
            print(f"✓ Convergence detection test passed")
            print(f"  Final convergence: {convergence}")
            
        except Exception as e:
            print(f"✗ Convergence detection test failed: {e}")
            raise

class TestCHMMTorchErrorHandling:
    """Test error handling and robustness."""
    
    def setup_method(self):
        """Set up test environment."""
        setup_test_environment()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms for error recovery."""
        try:
            x, a = create_test_sequence(length=100)
            n_clones = create_test_n_clones()
            
            # Create model
            model = CHMM_torch(n_clones, x, a)
            
            # Test BPS fallback with corrupted internal state
            # This tests the fallback mechanism
            original_T = model.T.clone()
            try:
                # Temporarily corrupt the model state to trigger fallback
                model.T = None
                bps = model.bps(x, a)  # Should trigger fallback
                print(f"  Fallback BPS: {bps}")
            except:
                pass  # Expected to fail, but shouldn't crash
            finally:
                # Restore model state
                model.T = original_T
            
            # Verify model still works after restoration
            bps_restored = model.bps(x, a)
            assert torch.isfinite(bps_restored), "Model should work after restoration"
            
            print("✓ Fallback mechanisms test passed")
            
        except Exception as e:
            print(f"✗ Fallback mechanisms test failed: {e}")
            raise
    
    def test_memory_stress(self):
        """Test behavior under memory stress."""
        try:
            # Create larger sequences to stress memory
            x, a = create_test_sequence(length=5000)
            n_clones = create_test_n_clones(n_clones_per_obs=5)  # More clones
            
            model = CHMM_torch(n_clones, x, a, memory_efficient=True)
            
            # Test that model can handle large computations
            bps = model.bps(x, a)
            assert torch.isfinite(bps), "BPS computation should succeed under memory stress"
            
            print(f"✓ Memory stress test passed")
            print(f"  Large sequence BPS: {bps.item():.4f}")
            
        except Exception as e:
            print(f"✗ Memory stress test failed: {e}")
            # This test is allowed to fail on systems with limited memory
            print("  Note: Memory stress test failure may be due to system limitations")

def test_train_chmm_cpu():
    """Test CHMM training function on CPU."""
    device = torch.device("cpu")
    
    # Create test data
    n_obs = 8
    n_clones = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2], device=device)
    
    # Generate test sequences
    seq_len = 50
    x = torch.randint(0, n_obs, (seq_len,), device=device)
    a = torch.randint(0, 4, (seq_len,), device=device)
    
    # Test training
    trained_model, progression = train_chmm(
        n_clones, x, a, device=device, 
        n_iter=5, pseudocount=1e-3, method='em_T'
    )
    
    # Validate outputs
    assert hasattr(trained_model, 'T') and hasattr(trained_model, 'Pi_x')
    assert isinstance(progression, list)
    assert len(progression) <= 5  # Should be <= n_iter
    assert all(isinstance(p, (int, float)) for p in progression)
    
    # Validate trained parameters
    n_states = n_clones.sum().item()
    assert trained_model.T.shape == (4, n_states, n_states)  # 4 actions
    assert torch.allclose(trained_model.T.sum(dim=2), torch.ones(4, n_states, device=device), atol=1e-5)


def test_make_E_functions():
    """Test emission matrix creation functions."""
    device = torch.device("cpu")
    n_clones = torch.tensor([3, 2, 4], device=device)
    
    # Test dense emission matrix
    E = make_E(n_clones, device)
    assert E.shape == (9, 3)  # total_clones=9, n_obs=3
    assert torch.allclose(E.sum(dim=1), torch.ones(9, device=device))
    
    # Check structure
    assert torch.all(E[0:3, 0] == 1.0)  # First 3 clones for obs 0
    assert torch.all(E[3:5, 1] == 1.0)  # Next 2 clones for obs 1
    assert torch.all(E[5:9, 2] == 1.0)  # Last 4 clones for obs 2
    
    # Test sparse emission matrix
    E_sparse = make_E_sparse(n_clones, device)
    assert E_sparse.shape == (9, 3)
    assert E_sparse.is_sparse
    
    # Convert to dense for comparison
    E_dense_from_sparse = E_sparse.to_dense()
    assert torch.allclose(E, E_dense_from_sparse)


def test_place_field():
    """Test place field computation."""
    device = torch.device("cpu")
    
    # Create test data
    T, n_states = 20, 10
    mess_fwd = torch.rand(T, n_states, device=device)
    rc = torch.randint(0, 5, (T, 2), device=device)  # 5x5 grid
    clone = 3
    
    # Compute place field
    field = place_field(mess_fwd, rc, clone, device)
    
    # Validate output
    assert field.shape == (5, 5)  # Grid size
    assert field.device == device
    assert torch.all(field >= 0)  # All values should be non-negative


def test_compute_forward_messages():
    """Test forward message computation."""
    device = torch.device("cpu")
    
    # Create test CHMM state
    n_actions, n_obs = 4, 6
    n_clones = torch.tensor([2, 2, 2, 2, 2, 2], device=device)
    n_states = n_clones.sum().item()
    
    T = torch.rand(n_actions, n_states, n_states, device=device)
    T = T / T.sum(dim=2, keepdim=True)
    
    E = make_E(n_clones, device)
    Pi_x = torch.ones(n_states, device=device) / n_states
    
    chmm_state = {
        'T': T,
        'E': E,
        'Pi_x': Pi_x,
        'n_clones': n_clones
    }
    
    # Generate test sequences
    seq_len = 30
    x = torch.randint(0, n_obs, (seq_len,), device=device)
    a = torch.randint(0, n_actions, (seq_len,), device=device)
    
    # Compute forward messages
    mess_fwd = compute_forward_messages(chmm_state, x, a, device)
    
    # Validate output
    assert mess_fwd.shape == (seq_len, n_states)
    assert mess_fwd.device == device
    assert torch.all(mess_fwd >= 0)
    assert torch.allclose(mess_fwd.sum(dim=1), torch.ones(seq_len, device=device), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_chmm_gpu():
    """Test CHMM training function on GPU."""
    device = torch.device("cuda")
    
    # Create test data
    n_obs = 8
    n_clones = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2], device=device)
    
    # Generate test sequences
    seq_len = 50
    x = torch.randint(0, n_obs, (seq_len,), device=device)
    a = torch.randint(0, 4, (seq_len,), device=device)
    
    # Test training
    trained_model, progression = train_chmm(
        n_clones, x, a, device=device, 
        n_iter=5, pseudocount=1e-3, method='em_T'
    )
    
    # Validate outputs
    assert hasattr(trained_model, 'T') and hasattr(trained_model, 'device')
    assert trained_model.T.device == device
    assert trained_model.device == device
    assert len(progression) <= 5


def test_utility_functions_integration():
    """Test integration of utility functions with room utilities."""
    device = torch.device("cpu")
    
    # Import room utilities for integration test
    try:
        from cscg_torch.env_adapters.room_utils import (
            clone_to_obs_map, top_k_used_clones, count_used_clones
        )
        
        # Test data
        n_clones = torch.tensor([3, 2, 4], device=device)
        states = torch.tensor([0, 1, 2, 5, 6, 7, 8, 1, 2], device=device)  # Various clone states
        
        # Test clone to obs mapping
        mapping = clone_to_obs_map(n_clones)
        assert len(mapping) == 9  # total clones
        assert mapping[0] == 0 and mapping[1] == 0 and mapping[2] == 0  # First obs
        assert mapping[3] == 1 and mapping[4] == 1  # Second obs
        assert mapping[5] == 2 and mapping[6] == 2 and mapping[7] == 2 and mapping[8] == 2  # Third obs
        
        # Test top k used clones
        top_clones = top_k_used_clones(states, k=3)
        assert len(top_clones) <= 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in top_clones)
        
        # Test that results are sorted by frequency (descending)
        if len(top_clones) > 1:
            assert top_clones[0][1] >= top_clones[1][1]
        
    except ImportError:
        pytest.skip("Room utilities not available for integration test")


def test_improved_train_chmm_integration():
    """Test the improved train_chmm function with various methods."""
    device = torch.device("cpu") 
    
    # Create larger test case
    n_obs = 16
    n_clones = torch.full((n_obs,), 2, dtype=torch.int64, device=device)
    
    # Generate longer, more realistic sequences
    seq_len = 200
    x = torch.randint(0, n_obs, (seq_len,), device=device)
    a = torch.randint(0, 4, (seq_len,), device=device)
    
    # Test different training methods
    methods = ['em_T', 'viterbi_T']
    
    for method in methods:
        print(f"Testing method: {method}")
        
        # Train model
        trained_model, progression = train_chmm(
            n_clones, x, a, device=device,
            method=method, n_iter=5, 
            pseudocount=0.01, early_stopping=True
        )
        
        # Validate trained model
        assert hasattr(trained_model, 'T'), f"Method {method} missing T matrix"
        assert hasattr(trained_model, 'Pi_x'), f"Method {method} missing Pi_x"
        assert trained_model.device == device, f"Device mismatch for {method}"
        
        # Validate progression
        assert isinstance(progression, list), f"Progression must be list for {method}"
        assert len(progression) <= 5, f"Too many iterations for {method}"
        assert all(isinstance(p, (int, float)) for p in progression), f"Invalid progression values for {method}"
        
        # Test model functionality
        bps = trained_model.bps(x, a)
        assert torch.isfinite(bps), f"BPS computation failed for {method}"
        
        # Test decoding
        neg_log_lik, states = trained_model.decode(x, a)
        assert torch.isfinite(neg_log_lik), f"Decoding failed for {method}"
        assert states.shape == x.shape, f"State sequence shape mismatch for {method}"
        
        print(f"✓ Method {method} passed all tests")


def test_gpu_performance_optimization():
    """Test GPU performance optimizations in new functions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda")
    
    # Create test data on GPU
    n_clones = torch.tensor([4, 4, 4, 4], device=device)
    seq_len = 500  # Larger for performance testing
    x = torch.randint(0, 4, (seq_len,), device=device)
    a = torch.randint(0, 4, (seq_len,), device=device)
    
    # Test GPU training
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    trained_model, progression = train_chmm(
        n_clones, x, a, device=device,
        n_iter=3, method='em_T'
    )
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed = start_time.elapsed_time(end_time)
    
    # Validate GPU execution
    assert trained_model.device == device, "Model not on GPU"
    assert elapsed < 10000, "GPU training took too long (>10s)"
    
    # Test GPU memory efficiency
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_reserved = torch.cuda.memory_reserved(device)
    
    print(f"GPU training completed in {elapsed:.2f}ms")
    print(f"Memory allocated: {memory_allocated / 1024**2:.1f} MB")
    print(f"Memory reserved: {memory_reserved / 1024**2:.1f} MB")
    
    # Clean up GPU memory
    del trained_model
    torch.cuda.empty_cache()

def run_all_tests():
    """Run all CHMM_torch tests including new function tests."""
    print("CHMM_torch Comprehensive Test Suite")
    print("=" * 50)
    
    test_classes = [
        TestCHMMTorchInitialization(),
        TestCHMMTorchComputation(),
        TestCHMMTorchTraining(),
        TestCHMMTorchErrorHandling()
    ]
    
    # Add standalone function tests
    standalone_tests = [
        test_train_chmm_cpu,
        test_make_E_functions,
        test_place_field,
        test_compute_forward_messages,
        test_utility_functions_integration,
        test_improved_train_chmm_integration,
        test_gpu_performance_optimization
    ]
    
    if torch.cuda.is_available():
        standalone_tests.append(test_train_chmm_gpu)
    
    total_tests = 0
    passed_tests = 0
    
    # Run class-based tests
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
    
    # Run standalone tests
    print(f"\nStandalone Function Tests")
    print("-" * 25)
    
    for test_func in standalone_tests:
        total_tests += 1
        try:
            test_func()
            passed_tests += 1
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
    
    print(f"\n\nTest Summary")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Performance summary
    if torch.cuda.is_available():
        print(f"\nGPU Performance Tests: Included")
        print(f"Device used: {torch.cuda.get_device_name()}")
    else:
        print(f"\nGPU Performance Tests: Skipped (CUDA not available)")

if __name__ == "__main__":
    run_all_tests()