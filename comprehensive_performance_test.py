#!/usr/bin/env python3
"""
Comprehensive granular performance test for optimized CHMM training.
Tests 15k steps of synthetic data on MPS to measure optimization impact.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

# Disable debug mode for performance testing
os.environ['CHMM_DEBUG'] = '0'

class ComprehensivePerformanceTest:
    """Comprehensive performance testing suite for CHMM optimizations."""
    
    def __init__(self):
        self.device = self._detect_device()
        self.results = {}
        self.timings = {}
        
        print(f"Performance Test Suite")
        print(f"Device: {self.device}")
        if self.device.type == 'mps':
            print("Testing optimizations on Apple Silicon MPS")
        elif self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name()
            print(f"GPU: {gpu_name}")
        
    def _detect_device(self):
        """Detect optimal device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _time_operation(self, name: str, func, *args, **kwargs):
        """Time an operation with proper synchronization."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        self.timings[name] = elapsed
        
        return result, elapsed
    
    def generate_synthetic_data(self, n_steps=15000):
        """Generate realistic synthetic data for testing."""
        print(f"\nGenerating {n_steps:,} steps of synthetic data...")
        
        # Realistic room navigation scenario
        room_size = 20  # 20x20 grid
        n_observations = room_size * room_size  # 400 observations
        n_actions = 4  # up, down, left, right
        clones_per_obs = 16  # typical for room navigation
        
        # Create n_clones tensor
        n_clones = torch.tensor([clones_per_obs] * n_observations, 
                               dtype=torch.int64, device=self.device)
        
        # Generate sequences with some structure (not purely random)
        torch.manual_seed(42)
        
        # Start from center and do random walk with some persistence
        current_pos = n_observations // 2
        x_seq = []
        a_seq = []
        
        for step in range(n_steps):
            x_seq.append(current_pos)
            
            # Choose action with some persistence (favor continuing in same direction)
            if step > 0 and np.random.random() < 0.7:  # 70% chance to continue
                action = a_seq[-1]
            else:
                action = np.random.randint(0, n_actions)
            
            a_seq.append(action)
            
            # Update position based on action (with boundary checking)
            row, col = divmod(current_pos, room_size)
            if action == 0 and row > 0:  # up
                current_pos -= room_size
            elif action == 1 and row < room_size - 1:  # down
                current_pos += room_size
            elif action == 2 and col > 0:  # left
                current_pos -= 1
            elif action == 3 and col < room_size - 1:  # right
                current_pos += 1
        
        x = torch.tensor(x_seq, dtype=torch.int64, device=self.device)
        a = torch.tensor(a_seq, dtype=torch.int64, device=self.device)
        
        print(f"Generated data:")
        print(f"  Sequence length: {len(x):,}")
        print(f"  Observations: {n_observations}")
        print(f"  Total states: {n_clones.sum().item():,}")
        print(f"  Actions: {n_actions}")
        print(f"  Unique positions visited: {len(torch.unique(x))}")
        
        return n_clones, x, a
    
    def test_matrix_operations(self):
        """Test individual matrix operation performance."""
        print(f"\n=== Matrix Operations Test ===")
        
        # Create test matrices similar to CHMM usage
        n_states = 6400  # 400 obs * 16 clones
        n_actions = 4
        batch_size = 1000
        
        T = torch.randn(n_actions, n_states, n_states, device=self.device)
        messages = torch.randn(batch_size, 80, device=self.device)  # typical clone group size
        
        # Test torch.mv vs torch.matmul
        def test_mv():
            for i in range(batch_size):
                result = torch.mv(T[i % n_actions, :80, :80], messages[i])
        
        def test_matmul():
            for i in range(batch_size):
                result = torch.matmul(T[i % n_actions, :80, :80], messages[i].unsqueeze(-1)).squeeze(-1)
        
        _, mv_time = self._time_operation("torch_mv", test_mv)
        _, matmul_time = self._time_operation("torch_matmul", test_matmul)
        
        speedup = mv_time / matmul_time
        print(f"torch.mv: {mv_time:.3f}s")
        print(f"torch.matmul: {matmul_time:.3f}s")
        print(f"matmul speedup: {speedup:.2f}x")
        
        return speedup
    
    def test_device_check_overhead(self, n_clones, x, a):
        """Test device check optimization impact."""
        print(f"\n=== Device Check Optimization Test ===")
        
        # Test with device checks enabled
        os.environ['CHMM_SKIP_DEVICE_CHECKS'] = '0'
        
        # Clear module cache
        if 'models.train_utils' in sys.modules:
            del sys.modules['models.train_utils']
        if 'models.chmm_torch' in sys.modules:
            del sys.modules['models.chmm_torch']
        
        def train_with_checks():
            from models.train_utils import train_chmm
            return train_chmm(n_clones, x, a, device=self.device, method='em_T', n_iter=1)
        
        (model1, prog1), time_with_checks = self._time_operation("with_device_checks", train_with_checks)
        
        # Test with device checks disabled
        os.environ['CHMM_SKIP_DEVICE_CHECKS'] = '1'
        
        # Clear module cache again
        if 'models.train_utils' in sys.modules:
            del sys.modules['models.train_utils']
        if 'models.chmm_torch' in sys.modules:
            del sys.modules['models.chmm_torch']
        
        def train_without_checks():
            from models.train_utils import train_chmm
            return train_chmm(n_clones, x, a, device=self.device, method='em_T', n_iter=1)
        
        (model2, prog2), time_without_checks = self._time_operation("without_device_checks", train_without_checks)
        
        # Calculate improvement
        if time_with_checks > time_without_checks:
            speedup = time_with_checks / time_without_checks
            improvement = ((time_with_checks - time_without_checks) / time_with_checks) * 100
        else:
            speedup = time_without_checks / time_with_checks
            improvement = -((time_without_checks - time_with_checks) / time_without_checks) * 100
        
        print(f"With device checks: {time_with_checks:.3f}s")
        print(f"Without device checks: {time_without_checks:.3f}s")
        print(f"Device check optimization: {speedup:.2f}x ({improvement:+.1f}%)")
        
        # Verify results are identical
        bps_diff = abs(prog1[-1] - prog2[-1])
        print(f"Result verification: BPS difference = {bps_diff:.8f}")
        
        return speedup, improvement
    
    def test_component_breakdown(self, n_clones, x, a):
        """Test individual component performance."""
        print(f"\n=== Component Breakdown Test ===")
        
        from models.chmm_torch import CHMM_torch
        from models.train_utils import forward, backward, updateC
        
        # Initialize model
        def init_model():
            return CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
        
        model, init_time = self._time_operation("model_initialization", init_model)
        
        # Test forward pass
        def forward_pass():
            T_tr = model.T.transpose(1, 2)
            workspace = {}
            return forward(T_tr, model.Pi_x, model.n_clones, x, a, self.device, 
                          store_messages=True, workspace=workspace)
        
        (log2_lik, mess_fwd), forward_time = self._time_operation("forward_pass", forward_pass)
        
        # Test backward pass
        def backward_pass():
            workspace = {}
            return backward(model.T, model.n_clones, x, a, self.device, workspace=workspace)
        
        mess_bwd, backward_time = self._time_operation("backward_pass", backward_pass)
        
        # Test count update
        def count_update():
            C = torch.zeros_like(model.T)
            workspace = {}
            updateC(C, model.T, model.n_clones, mess_fwd, mess_bwd, x, a, self.device, workspace=workspace)
            return C
        
        C, update_time = self._time_operation("count_update", count_update)
        
        total_component_time = init_time + forward_time + backward_time + update_time
        
        print(f"Model initialization: {init_time:.3f}s ({init_time/total_component_time*100:.1f}%)")
        print(f"Forward pass: {forward_time:.3f}s ({forward_time/total_component_time*100:.1f}%)")
        print(f"Backward pass: {backward_time:.3f}s ({backward_time/total_component_time*100:.1f}%)")
        print(f"Count update: {update_time:.3f}s ({update_time/total_component_time*100:.1f}%)")
        print(f"Total components: {total_component_time:.3f}s")
        
        # Calculate per-step costs
        per_step_forward = forward_time / len(x) * 1000  # ms per step
        per_step_backward = backward_time / len(x) * 1000
        
        print(f"Per-step costs:")
        print(f"  Forward: {per_step_forward:.3f}ms/step")
        print(f"  Backward: {per_step_backward:.3f}ms/step")
        
        # Extrapolate to 150k steps
        total_per_step = (forward_time + backward_time) / len(x)
        estimated_150k = total_per_step * 150000
        
        print(f"Estimated time for 150k steps: {estimated_150k:.1f}s ({estimated_150k/60:.1f} minutes)")
        
        return {
            'init_time': init_time,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'update_time': update_time,
            'per_step_ms': total_per_step * 1000,
            'estimated_150k_seconds': estimated_150k
        }
    
    def test_full_training_performance(self, n_clones, x, a):
        """Test complete training iteration performance."""
        print(f"\n=== Full Training Performance Test ===")
        
        from models.train_utils import train_chmm
        
        # Single iteration test
        def single_iteration():
            return train_chmm(n_clones, x, a, device=self.device, method='em_T', n_iter=1, seed=42)
        
        (model, progression), single_time = self._time_operation("single_training_iteration", single_iteration)
        
        # Multiple iteration test (3 iterations)
        def multiple_iterations():
            return train_chmm(n_clones, x, a, device=self.device, method='em_T', n_iter=3, seed=42)
        
        (model_multi, prog_multi), multi_time = self._time_operation("three_training_iterations", multiple_iterations)
        
        # Calculate metrics
        per_iteration_time = multi_time / 3
        steps_per_second = len(x) / single_time
        
        print(f"Single iteration: {single_time:.3f}s")
        print(f"Three iterations: {multi_time:.3f}s")
        print(f"Average per iteration: {per_iteration_time:.3f}s")
        print(f"Processing speed: {steps_per_second:.0f} steps/second")
        
        # Final BPS
        final_bps = progression[-1]
        print(f"Final BPS: {final_bps:.6f}")
        
        # Extrapolate to 150k steps with 50 iterations
        estimated_total_150k = (per_iteration_time * 50) * (150000 / len(x))
        
        print(f"Estimated 150k steps, 50 iterations: {estimated_total_150k:.1f}s ({estimated_total_150k/60:.1f} minutes)")
        
        return {
            'single_iteration_time': single_time,
            'per_iteration_time': per_iteration_time,
            'steps_per_second': steps_per_second,
            'final_bps': final_bps,
            'estimated_150k_50iter_seconds': estimated_total_150k
        }
    
    def test_memory_usage(self, n_clones, x, a):
        """Test memory usage patterns."""
        print(f"\n=== Memory Usage Test ===")
        
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            
            from models.train_utils import train_chmm
            model, progression = train_chmm(n_clones, x, a, device=self.device, method='em_T', n_iter=1)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            final_memory = torch.cuda.memory_allocated() / 1024**3
            
            print(f"Initial memory: {initial_memory:.2f}GB")
            print(f"Peak memory: {peak_memory:.2f}GB")
            print(f"Final memory: {final_memory:.2f}GB")
            print(f"Memory increase: {peak_memory - initial_memory:.2f}GB")
            
            return {
                'initial_memory_gb': initial_memory,
                'peak_memory_gb': peak_memory,
                'memory_increase_gb': peak_memory - initial_memory
            }
        else:
            print("Memory profiling only available on CUDA")
            return {'memory_profiling': 'not_available'}
    
    def generate_comprehensive_report(self):
        """Generate comprehensive performance report."""
        print(f"\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\nDevice: {self.device}")
        print(f"Test data: 15,000 steps")
        
        # Optimization impact summary
        print(f"\nOptimization Impact:")
        if 'torch_mv' in self.timings and 'torch_matmul' in self.timings:
            matmul_speedup = self.timings['torch_mv'] / self.timings['torch_matmul']
            print(f"  Matrix operation (torch.matmul vs torch.mv): {matmul_speedup:.2f}x speedup")
        
        if 'with_device_checks' in self.timings and 'without_device_checks' in self.timings:
            device_speedup = self.timings['with_device_checks'] / self.timings['without_device_checks']
            print(f"  Device check elimination: {device_speedup:.2f}x speedup")
        
        # Performance breakdown
        print(f"\nPerformance Breakdown:")
        component_times = [
            ('Model Init', self.timings.get('model_initialization', 0)),
            ('Forward Pass', self.timings.get('forward_pass', 0)),
            ('Backward Pass', self.timings.get('backward_pass', 0)),
            ('Count Update', self.timings.get('count_update', 0)),
            ('Full Iteration', self.timings.get('single_training_iteration', 0))
        ]
        
        for name, time_val in component_times:
            if time_val > 0:
                print(f"  {name}: {time_val:.3f}s")
        
        # Bottleneck analysis
        print(f"\nBottleneck Analysis:")
        sorted_times = sorted([(k, v) for k, v in self.timings.items() if v > 0.1], 
                             key=lambda x: x[1], reverse=True)
        
        for i, (operation, time_val) in enumerate(sorted_times[:5]):
            print(f"  {i+1}. {operation}: {time_val:.3f}s")
        
        # Scaling predictions
        if 'single_training_iteration' in self.timings:
            single_time = self.timings['single_training_iteration']
            scaling_factor = 150000 / 15000  # 150k vs 15k steps
            
            print(f"\nScaling Predictions for 150k steps:")
            print(f"  Single iteration (15k): {single_time:.3f}s")
            print(f"  Predicted single iteration (150k): {single_time * scaling_factor:.1f}s")
            print(f"  Predicted 50 iterations (150k): {single_time * scaling_factor * 50:.1f}s ({single_time * scaling_factor * 50 / 60:.1f} minutes)")
        
        # Recommendations
        print(f"\nRecommendations:")
        if 'forward_pass' in self.timings and 'backward_pass' in self.timings:
            total_fb = self.timings['forward_pass'] + self.timings['backward_pass']
            total_time = self.timings.get('single_training_iteration', total_fb)
            fb_percent = (total_fb / total_time) * 100 if total_time > 0 else 0
            
            if fb_percent > 80:
                print(f"  Forward/backward passes dominate ({fb_percent:.1f}%) - focus on algorithmic improvements")
            elif fb_percent > 60:
                print(f"  Forward/backward passes significant ({fb_percent:.1f}%) - consider chunking or vectorization")
            else:
                print(f"  Balanced computation profile - optimizations are working well")
        
        return self.timings
    
    def run_comprehensive_test(self):
        """Run the complete performance test suite."""
        print("Starting Comprehensive Performance Test...")
        print("This will test all optimization impacts on 15k synthetic data")
        
        # Generate test data
        n_clones, x, a = self.generate_synthetic_data(15000)
        
        # Run all tests
        matrix_speedup = self.test_matrix_operations()
        device_speedup, device_improvement = self.test_device_check_overhead(n_clones, x, a)
        component_results = self.test_component_breakdown(n_clones, x, a)
        training_results = self.test_full_training_performance(n_clones, x, a)
        memory_results = self.test_memory_usage(n_clones, x, a)
        
        # Generate final report
        timings = self.generate_comprehensive_report()
        
        return {
            'matrix_speedup': matrix_speedup,
            'device_speedup': device_speedup,
            'component_results': component_results,
            'training_results': training_results,
            'memory_results': memory_results,
            'all_timings': timings
        }

def main():
    """Main test function."""
    tester = ComprehensivePerformanceTest()
    results = tester.run_comprehensive_test()
    
    print(f"\nTest completed successfully!")
    print(f"Key metrics:")
    print(f"  Matrix operation speedup: {results['matrix_speedup']:.2f}x")
    print(f"  Device check optimization: {results['device_speedup']:.2f}x")
    print(f"  Estimated 150k/50iter time: {results['training_results']['estimated_150k_50iter_seconds']/60:.1f} minutes")
    
    return results

if __name__ == "__main__":
    results = main()