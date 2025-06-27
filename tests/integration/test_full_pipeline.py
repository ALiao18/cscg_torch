"""
Integration Tests for Full CSCG Pipeline

End-to-end tests combining room environments, CHMM models, and training:
- Full training pipeline with room navigation
- Performance benchmarks
- Memory usage monitoring
- GPU optimization validation
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cscg_torch.models.chmm_torch import CHMM_torch
from cscg_torch.env_adapters.room_utils import create_room_adapter, get_room_n_clones
from tests.test_config import (
    TestConfig, setup_test_environment, cleanup_test_environment,
    load_test_room, measure_performance, check_gpu_memory,
    gpu_test, slow_test
)

class TestFullPipeline:
    """Test complete CSCG training pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        setup_test_environment()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    def test_basic_pipeline(self):
        """Test basic end-to-end pipeline."""
        try:
            print("\n  Setting up environment...")
            
            # Load test room and create adapter
            room = load_test_room(TestConfig.ROOM_SIZE)
            adapter = create_room_adapter(room, adapter_type="torch", seed=TestConfig.SEED)
            
            # Generate training sequence
            print("Generating training sequence...")
            x_seq, a_seq = adapter.generate_sequence(TestConfig.SEQUENCE_LENGTH)
            
            # Convert to tensors
            x = torch.tensor(x_seq, dtype=torch.int64)
            a = torch.tensor(a_seq, dtype=torch.int64)
            n_clones = get_room_n_clones(n_clones_per_obs=TestConfig.N_CLONES_PER_OBS)
            
            print(f"  Generated sequence: {len(x)} steps")
            print(f"  Obs range: {x.min().item()} to {x.max().item()}")
            print(f"  Action range: {a.min().item()} to {a.max().item()}")
            
            # Initialize CHMM model
            print("  Initializing CHMM model...")
            model = CHMM_torch(n_clones, x, a, pseudocount=0.01, seed=TestConfig.SEED)
            
            # Compute initial likelihood
            initial_bps = model.bps(x, a, reduce=True).item()
            print(f"  Initial BPS: {initial_bps:.4f}")
            
            # Train model
            print("  Training model...")
            convergence, training_time = measure_performance(
                model.learn_em_T, x, a, 
                n_iter=TestConfig.EM_ITERATIONS, 
                term_early=True
            )
            
            # Compute final likelihood
            final_bps = model.bps(x, a, reduce=True).item()
            improvement = initial_bps - final_bps
            
            # Validate results
            assert len(convergence) > 0, "Training produced no convergence history"
            assert improvement >= -1.0, "Model performance severely degraded"  # Allow some numerical noise
            
            print(f"  ✓ Basic pipeline test passed")
            print(f"    Training time: {training_time:.2f}s")
            print(f"    Final BPS: {final_bps:.4f}")
            print(f"    Improvement: {improvement:.4f}")
            print(f"    Iterations: {len(convergence)}")
            
            return {
                'training_time': training_time,
                'initial_bps': initial_bps,
                'final_bps': final_bps,
                'improvement': improvement,
                'iterations': len(convergence)
            }
        except Exception as e:
            print(f"  ✗ Basic pipeline test failed: {e}")
            raise
    
    def test_multiple_room_sizes(self):
        """Test pipeline with different room sizes."""
        try:
            test_sizes = [5, 10]  # Use smaller sizes for speed
            results = {}
            
            for size in test_sizes:
                print(f"\n  Testing {size}x{size} room...")
                
                # Load room and create adapter
                room = load_test_room(size)
                adapter = create_room_adapter(room, adapter_type="torch", seed=TestConfig.SEED)
                
                # Generate shorter sequence for larger rooms
                seq_length = max(500, 2000 // size)  # Adaptive sequence length
                x_seq, a_seq = adapter.generate_sequence(seq_length)
                
                # Convert to tensors
                x = torch.tensor(x_seq, dtype=torch.int64)
                a = torch.tensor(a_seq, dtype=torch.int64)
                n_clones = get_room_n_clones(n_clones_per_obs=2)
                
                # Train model
                model = CHMM_torch(n_clones, x, a, pseudocount=0.01)
                initial_bps = model.bps(x, a, reduce=True).item()
                
                convergence, training_time = measure_performance(
                    model.learn_em_T, x, a, 
                    n_iter=5,  # Shorter training for speed
                    term_early=True
                )
                
                final_bps = model.bps(x, a, reduce=True).item()
                
                results[size] = {
                    'seq_length': len(x),
                    'training_time': training_time,
                    'initial_bps': initial_bps,
                    'final_bps': final_bps,
                    'iterations': len(convergence)
                }
                
                print(f"    ✓ {size}x{size} room: {len(x)} steps, {training_time:.2f}s")
            
            print(f"  ✓ Multiple room sizes test passed")
            for size, result in results.items():
                print(f"    {size}x{size}: {result['seq_length']} steps, {result['training_time']:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"  ✗ Multiple room sizes test failed: {e}")
            raise
    
    @gpu_test
    def test_gpu_pipeline_performance(self):
        """Test GPU-specific pipeline performance."""
        try:
            print("\n  Testing GPU pipeline performance...")
            
            # Monitor initial GPU memory
            memory_before = check_gpu_memory()
            
            # Set up pipeline with GPU optimization
            room = load_test_room(TestConfig.ROOM_SIZE)
            adapter = create_room_adapter(room, adapter_type="torch")
            
            # Generate training data
            x_seq, a_seq = adapter.generate_sequence(TestConfig.SEQUENCE_LENGTH)
            x = torch.tensor(x_seq, dtype=torch.int64)
            a = torch.tensor(a_seq, dtype=torch.int64)
            n_clones = get_room_n_clones(n_clones_per_obs=TestConfig.N_CLONES_PER_OBS)
            
            # Initialize model with GPU optimizations
            model = CHMM_torch(
                n_clones, x, a, 
                enable_mixed_precision=True,
                memory_efficient=True,
                seed=TestConfig.SEED
            )
            
            # Verify GPU placement
            assert model.device.type == 'cuda', "Model should be on GPU"
            assert model.T.device.type == 'cuda', "Transition matrix should be on GPU"
            
            # Train with performance monitoring
            convergence, training_time = measure_performance(
                model.learn_em_T, x, a,
                n_iter=TestConfig.EM_ITERATIONS
            )
            
            # Monitor final GPU memory
            memory_after = check_gpu_memory()
            
            # Performance validation
            assert training_time < 60, f"GPU training too slow: {training_time:.2f}s"
            assert len(convergence) > 0, "Training produced no results"
            
            print(f"  ✓ GPU pipeline performance test passed")
            print(f"    Training time: {training_time:.2f}s")
            print(f"    Memory before: {memory_before['allocated_mb']:.1f} MB")
            print(f"    Memory after: {memory_after['allocated_mb']:.1f} MB")
            print(f"    Peak memory: {memory_after['max_allocated_mb']:.1f} MB")
            print(f"    Iterations: {len(convergence)}")
            
            return {
                'training_time': training_time,
                'memory_before': memory_before['allocated_mb'],
                'memory_after': memory_after['allocated_mb'],
                'peak_memory': memory_after['max_allocated_mb'],
                'iterations': len(convergence)
            }
            
        except Exception as e:
            print(f"  ✗ GPU pipeline performance test failed: {e}")
            raise
    
    def test_viterbi_decoding_pipeline(self):
        """Test pipeline with Viterbi decoding."""
        try:
            print("\n  Testing Viterbi decoding pipeline...")
            
            # Set up basic pipeline
            room = load_test_room(TestConfig.ROOM_SIZE)
            adapter = create_room_adapter(room, adapter_type="torch")
            
            # Generate test sequence
            x_seq, a_seq = adapter.generate_sequence(1000)  # Shorter for speed
            x = torch.tensor(x_seq, dtype=torch.int64)
            a = torch.tensor(a_seq, dtype=torch.int64)
            n_clones = get_room_n_clones(n_clones_per_obs=2)
            
            # Train model
            model = CHMM_torch(n_clones, x, a, pseudocount=0.01)
            convergence = model.learn_em_T(x, a, n_iter=5)
            
            # Test Viterbi decoding
            print("    Running Viterbi decoding...")
            neg_log_lik, states, decoding_time = measure_performance(
                model.decode, x, a
            )
            
            # Validate decoding results
            assert isinstance(neg_log_lik, torch.Tensor), "Negative log-likelihood must be tensor"
            assert neg_log_lik.ndim == 0, "Negative log-likelihood must be scalar"
            assert torch.isfinite(neg_log_lik), "Negative log-likelihood must be finite"
            
            assert isinstance(states, torch.Tensor), "States must be tensor"
            assert states.shape == (len(x),), f"States shape mismatch: {states.shape}"
            assert states.dtype in [torch.int64, torch.long], "States must be integer type"
            
            # Check state validity
            max_state = n_clones.sum().item() - 1
            assert torch.all(states >= 0), "All states must be non-negative"
            assert torch.all(states <= max_state), "All states must be valid"
            
            # Compare with BPS
            bps = model.bps(x, a, reduce=True)
            
            print(f"  ✓ Viterbi decoding pipeline test passed")
            print(f"    Decoding time: {decoding_time:.3f}s")
            print(f"    MAP log-likelihood: {-neg_log_lik.item():.4f}")
            print(f"    BPS: {bps.item():.4f}")
            print(f"    State range: {states.min().item()} to {states.max().item()}")
            
            return {
                'decoding_time': decoding_time,
                'map_log_lik': -neg_log_lik.item(),
                'bps': bps.item(),
                'state_range': (states.min().item(), states.max().item())
            }
            
        except Exception as e:
            print(f"  ✗ Viterbi decoding pipeline test failed: {e}")
            raise

class TestPerformanceBenchmarks:
    """Performance benchmarks for different configurations."""
    
    def setup_method(self):
        """Set up test environment."""
        setup_test_environment()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_test_environment()
    
    @slow_test
    def test_scalability_benchmark(self):
        """Test scalability with different sequence lengths."""
        try:
            print("\n  Running scalability benchmark...")
            
            # Test different sequence lengths
            sequence_lengths = [500, 1000, 2000, 4000]
            benchmark_results = {}
            
            room = load_test_room(TestConfig.ROOM_SIZE)
            adapter = create_room_adapter(room, adapter_type="torch")
            
            for seq_len in sequence_lengths:
                print(f"    Testing sequence length: {seq_len}")
                
                # Generate sequence
                x_seq, a_seq = adapter.generate_sequence(seq_len)
                x = torch.tensor(x_seq[:seq_len], dtype=torch.int64)  # Ensure exact length
                a = torch.tensor(a_seq[:seq_len], dtype=torch.int64)
                n_clones = get_room_n_clones(n_clones_per_obs=2)
                
                # Initialize and train
                model = CHMM_torch(n_clones, x, a, pseudocount=0.01)
                
                # Time training
                convergence, training_time = measure_performance(
                    model.learn_em_T, x, a, n_iter=3  # Short training for benchmark
                )
                
                # Time inference
                bps, inference_time = measure_performance(
                    model.bps, x, a, reduce=True
                )
                
                benchmark_results[seq_len] = {
                    'actual_length': len(x),
                    'training_time': training_time,
                    'inference_time': inference_time,
                    'bps': bps.item(),
                    'iterations': len(convergence)
                }
                
                print(f"      Training: {training_time:.3f}s, Inference: {inference_time:.3f}s")
            
            print(f"  ✓ Scalability benchmark completed")
            for seq_len, result in benchmark_results.items():
                print(f"    {seq_len}: train={result['training_time']:.3f}s, infer={result['inference_time']:.3f}s")
            
            return benchmark_results
            
        except Exception as e:
            print(f"  ✗ Scalability benchmark failed: {e}")
            raise
    
    def test_memory_usage_benchmark(self):
        """Test memory usage patterns."""
        try:
            print("\n  Running memory usage benchmark...")
            
            # Monitor memory throughout pipeline
            memory_points = []
            
            def record_memory(label):
                if TestConfig.USE_GPU:
                    memory_info = check_gpu_memory()
                    memory_points.append((label, memory_info['allocated_mb']))
                    print(f"    {label}: {memory_info['allocated_mb']:.1f} MB")
            
            record_memory("Initial")
            
            # Set up pipeline
            room = load_test_room(TestConfig.ROOM_SIZE)
            adapter = create_room_adapter(room, adapter_type="torch")
            record_memory("After room load")
            
            # Generate data
            x_seq, a_seq = adapter.generate_sequence(TestConfig.SEQUENCE_LENGTH)
            x = torch.tensor(x_seq, dtype=torch.int64)
            a = torch.tensor(a_seq, dtype=torch.int64)
            n_clones = get_room_n_clones(n_clones_per_obs=TestConfig.N_CLONES_PER_OBS)
            record_memory("After data generation")
            
            # Initialize model
            model = CHMM_torch(n_clones, x, a, memory_efficient=True)
            record_memory("After model initialization")
            
            # Train
            convergence = model.learn_em_T(x, a, n_iter=3)
            record_memory("After training")
            
            # Clean up
            del model, x, a, n_clones
            if TestConfig.USE_GPU:
                torch.cuda.empty_cache()
            record_memory("After cleanup")
            
            print(f"  ✓ Memory usage benchmark completed")
            
            return memory_points
            
        except Exception as e:
            print(f"  ✗ Memory usage benchmark failed: {e}")
            raise

def run_all_tests():
    """Run all integration tests."""
    print("CSCG Integration Test Suite")
    print("=" * 50)
    
    test_classes = [
        TestFullPipeline(),
        TestPerformanceBenchmarks()
    ]
    
    total_tests = 0
    passed_tests = 0
    test_results = {}
    
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
                result = method()
                passed_tests += 1
                
                # Store result if returned
                if result is not None:
                    test_results[method_name] = result
                
                # Tear down test
                if hasattr(test_class, 'teardown_method'):
                    test_class.teardown_method()
                    
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                # Continue with other tests
                if hasattr(test_class, 'teardown_method'):
                    test_class.teardown_method()
    
    print(f"\n\nIntegration Test Summary")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Print performance summary if available
    if test_results:
        print(f"\nPerformance Summary")
        print("-" * 20)
        for test_name, result in test_results.items():
            if isinstance(result, dict) and 'training_time' in result:
                print(f"{test_name}: {result['training_time']:.2f}s training")

if __name__ == "__main__":
    run_all_tests()