#!/usr/bin/env python3
"""
Performance profiling script for CHMM training bottleneck analysis.
Generate 150k steps and profile 1 training iteration with granular timing.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import psutil
import gc

# Disable debug mode for performance testing
os.environ['CHMM_DEBUG'] = '0'

# Set up for A100 optimization (but will work on MPS for testing)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

try:
    from cscg_torch import create_room_adapter, CHMM_torch
    from cscg_torch.models.train_utils import forward, backward, updateC, train_chmm
    from cscg_torch.utils import load_room_data, detect_optimal_device
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    exit(1)

class PerformanceProfiler:
    """Granular performance profiler for CHMM training operations."""
    
    def __init__(self, device=None):
        self.device = device or detect_optimal_device()
        self.timings = {}
        self.memory_usage = {}
        self.theoretical_times = {}
        
        # A100 theoretical performance (TFLOPS)
        self.a100_fp32_tflops = 19.5  # A100 FP32 performance
        self.a100_memory_bandwidth = 1555  # GB/s
        
        print(f"Profiler initialized on {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
    def time_operation(self, name: str, func, *args, **kwargs):
        """Time a specific operation with GPU synchronization."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        self.timings[name] = elapsed
        
        # Memory usage
        if self.device.type == 'cuda':
            self.memory_usage[name] = torch.cuda.memory_allocated() / 1024**3
        
        return result
    
    def estimate_theoretical_time(self, operation: str, tensor_ops: int, memory_ops: int):
        """Estimate theoretical time for operation on A100."""
        # Compute time (assuming FP32 operations)
        compute_time = tensor_ops / (self.a100_fp32_tflops * 1e12)
        
        # Memory time (GB transferred)
        memory_time = memory_ops / (self.a100_memory_bandwidth * 1e9)
        
        # Take max (compute or memory bound)
        theoretical_time = max(compute_time, memory_time)
        self.theoretical_times[operation] = theoretical_time
        
        return theoretical_time
    
    def generate_test_data(self, n_steps: int = 150000):
        """Generate test data for performance profiling."""
        print(f"Generating {n_steps} step sequence...")
        
        # Create room adapter
        room_data = load_room_data("room_20x20") if hasattr(load_room_data, '__call__') else self._create_default_room()
        adapter = create_room_adapter(room_data)
        
        # Generate sequence
        start_time = time.time()
        x_seq, a_seq = adapter.generate_sequence_gpu(n_steps, device=self.device)
        gen_time = time.time() - start_time
        
        print(f"Data generation: {gen_time:.2f}s ({n_steps/gen_time:.0f} steps/sec)")
        
        # Get n_clones
        n_clones = adapter.get_n_clones().to(self.device)
        
        return n_clones, x_seq, a_seq
    
    def _create_default_room(self):
        """Create default room data if load fails."""
        return {
            'room_structure': np.ones((20, 20), dtype=int),
            'n_clones': torch.tensor([16] * 400, dtype=torch.int64)
        }
    
    def profile_chmm_initialization(self, n_clones, x, a):
        """Profile CHMM model initialization."""
        print("\n=== CHMM Initialization Profiling ===")
        
        def init_chmm():
            return CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
        
        model = self.time_operation("chmm_init", init_chmm)
        
        # Theoretical: mostly memory allocation
        n_states = n_clones.sum().item()
        n_actions = a.max().item() + 1
        
        # Memory for T matrix [n_actions, n_states, n_states]
        T_memory = n_actions * n_states * n_states * 4  # 4 bytes per float32
        init_memory_gb = T_memory / 1024**3
        
        self.estimate_theoretical_time("chmm_init", 
                                     tensor_ops=n_states * n_actions,  # initialization ops
                                     memory_ops=init_memory_gb)
        
        return model
    
    def profile_forward_pass(self, model, x, a):
        """Profile forward pass with detailed breakdown."""
        print("\n=== Forward Pass Profiling ===")
        
        # Get model parameters
        T_tr = model.T.transpose(1, 2)
        Pi_x = model.Pi_x
        n_clones = model.n_clones
        
        # Profile individual components
        workspace = {}
        
        def forward_pass():
            return forward(T_tr, Pi_x, n_clones, x, a, self.device, 
                         store_messages=True, workspace=workspace)
        
        log2_lik, mess_fwd = self.time_operation("forward_pass", forward_pass)
        
        # Detailed timing breakdown
        seq_len = len(x)
        n_states = Pi_x.shape[0]
        n_actions = a.max().item() + 1
        
        # Theoretical computation:
        # - Each timestep: matrix-vector multiplication
        # - T_tr slice: [n_j_clones, n_i_clones] * [n_i_clones] = [n_j_clones]
        # - Average case: sqrt(n_states) operations per timestep
        avg_ops_per_step = int(np.sqrt(n_states))
        total_ops = seq_len * avg_ops_per_step * avg_ops_per_step
        
        # Memory: reading T_tr slices + writing messages
        memory_per_step = avg_ops_per_step * avg_ops_per_step * 4  # float32
        total_memory_gb = (seq_len * memory_per_step) / 1024**3
        
        self.estimate_theoretical_time("forward_pass", 
                                     tensor_ops=total_ops,
                                     memory_ops=total_memory_gb)
        
        return log2_lik, mess_fwd
    
    def profile_backward_pass(self, model, x, a):
        """Profile backward pass."""
        print("\n=== Backward Pass Profiling ===")
        
        T = model.T
        n_clones = model.n_clones
        workspace = {}
        
        def backward_pass():
            return backward(T, n_clones, x, a, self.device, workspace=workspace)
        
        mess_bwd = self.time_operation("backward_pass", backward_pass)
        
        # Similar theoretical analysis as forward pass
        seq_len = len(x)
        n_states = n_clones.sum().item()
        avg_ops_per_step = int(np.sqrt(n_states))
        total_ops = seq_len * avg_ops_per_step * avg_ops_per_step
        total_memory_gb = (seq_len * avg_ops_per_step * avg_ops_per_step * 4) / 1024**3
        
        self.estimate_theoretical_time("backward_pass", 
                                     tensor_ops=total_ops,
                                     memory_ops=total_memory_gb)
        
        return mess_bwd
    
    def profile_update_counts(self, model, x, a, mess_fwd, mess_bwd):
        """Profile count matrix update."""
        print("\n=== Update Counts Profiling ===")
        
        T = model.T
        n_clones = model.n_clones
        C = torch.zeros_like(T)
        workspace = {}
        
        def update_counts():
            updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a, self.device, workspace=workspace)
            return C
        
        C = self.time_operation("update_counts", update_counts)
        
        # Theoretical: outer products for each timestep
        seq_len = len(x)
        n_states = n_clones.sum().item()
        avg_ops_per_step = int(np.sqrt(n_states))
        
        # Each timestep: outer product of messages
        total_ops = seq_len * avg_ops_per_step * avg_ops_per_step
        total_memory_gb = (seq_len * avg_ops_per_step * avg_ops_per_step * 4) / 1024**3
        
        self.estimate_theoretical_time("update_counts", 
                                     tensor_ops=total_ops,
                                     memory_ops=total_memory_gb)
        
        return C
    
    def profile_full_training_iteration(self, n_clones, x, a):
        """Profile a complete training iteration."""
        print("\n=== Full Training Iteration Profiling ===")
        
        def single_iteration():
            return train_chmm(n_clones, x, a, device=self.device, 
                            method='em_T', n_iter=1, pseudocount=0.01, seed=42)
        
        model, progression = self.time_operation("full_iteration", single_iteration)
        
        # Theoretical: sum of all components
        seq_len = len(x)
        n_states = n_clones.sum().item()
        n_actions = a.max().item() + 1
        
        # Full iteration = forward + backward + update + normalization
        total_ops = seq_len * n_states * 4  # rough estimate
        total_memory_gb = (n_actions * n_states * n_states * 4) / 1024**3
        
        self.estimate_theoretical_time("full_iteration", 
                                     tensor_ops=total_ops,
                                     memory_ops=total_memory_gb)
        
        return model, progression
    
    def analyze_assert_overhead(self, model, x, a, n_samples=100):
        """Measure assert statement overhead."""
        print("\n=== Assert Statement Overhead Analysis ===")
        
        # Enable debug mode
        os.environ['CHMM_DEBUG'] = '1'
        
        def forward_with_asserts():
            return forward(model.T.transpose(1, 2), model.Pi_x, model.n_clones, 
                         x[:n_samples], a[:n_samples], self.device)
        
        time_with_asserts = self.time_operation("forward_with_asserts", forward_with_asserts)
        
        # Disable debug mode
        os.environ['CHMM_DEBUG'] = '0'
        
        def forward_without_asserts():
            return forward(model.T.transpose(1, 2), model.Pi_x, model.n_clones, 
                         x[:n_samples], a[:n_samples], self.device)
        
        time_without_asserts = self.time_operation("forward_without_asserts", forward_without_asserts)
        
        assert_overhead = time_with_asserts[0] - time_without_asserts[0]
        overhead_percent = (assert_overhead / time_without_asserts[0]) * 100
        
        print(f"Assert overhead: {assert_overhead:.4f}s ({overhead_percent:.1f}%)")
        
        return assert_overhead
    
    def check_device_transfers(self, model, x, a):
        """Check for unnecessary CPU-GPU transfers."""
        print("\n=== Device Transfer Analysis ===")
        
        # Check tensor devices
        tensors_to_check = [
            ("model.T", model.T),
            ("model.Pi_x", model.Pi_x),
            ("model.n_clones", model.n_clones),
            ("x", x),
            ("a", a)
        ]
        
        device_mismatches = []
        for name, tensor in tensors_to_check:
            if tensor.device != self.device:
                device_mismatches.append(f"{name}: {tensor.device} -> {self.device}")
        
        if device_mismatches:
            print("❌ Device mismatches found:")
            for mismatch in device_mismatches:
                print(f"   {mismatch}")
        else:
            print("✅ All tensors on correct device")
        
        return len(device_mismatches) == 0
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nDevice: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB peak")
        
        print("\nTIMING ANALYSIS:")
        print("-"*40)
        
        total_actual = 0
        total_theoretical = 0
        
        for operation in self.timings:
            actual = self.timings[operation]
            theoretical = self.theoretical_times.get(operation, 0)
            efficiency = (theoretical / actual * 100) if actual > 0 else 0
            
            print(f"{operation:20s}: {actual:8.3f}s (theoretical: {theoretical:8.3f}s, efficiency: {efficiency:5.1f}%)")
            
            total_actual += actual
            total_theoretical += theoretical
        
        print("-"*40)
        print(f"{'TOTAL':20s}: {total_actual:8.3f}s (theoretical: {total_theoretical:8.3f}s)")
        
        overall_efficiency = (total_theoretical / total_actual * 100) if total_actual > 0 else 0
        print(f"Overall GPU efficiency: {overall_efficiency:.1f}%")
        
        print("\nBOTTLENECK ANALYSIS:")
        print("-"*40)
        
        # Sort by actual time to identify bottlenecks
        bottlenecks = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for i, (op, time) in enumerate(bottlenecks[:5]):
            percent = (time / total_actual) * 100
            print(f"{i+1}. {op}: {time:.3f}s ({percent:.1f}% of total)")
        
        print("\nOPTIMIZATION RECOMMENDATIONS:")
        print("-"*40)
        
        if overall_efficiency < 30:
            print("❌ Very low GPU efficiency - major optimizations needed")
        elif overall_efficiency < 60:
            print("⚠️  Moderate GPU efficiency - some optimizations possible")
        else:
            print("✅ Good GPU efficiency")
        
        # Specific recommendations based on bottlenecks
        worst_operation = bottlenecks[0][0]
        worst_efficiency = (self.theoretical_times.get(worst_operation, 0) / 
                          self.timings[worst_operation] * 100)
        
        if worst_efficiency < 20:
            print(f"🔥 {worst_operation} has very low efficiency ({worst_efficiency:.1f}%) - priority optimization target")
        
        return {
            'total_time': total_actual,
            'theoretical_time': total_theoretical,
            'efficiency': overall_efficiency,
            'bottlenecks': bottlenecks
        }

def main():
    """Main profiling function."""
    print("Starting CHMM Performance Profiling...")
    print("Target: 150k steps, 1 training iteration")
    
    # Initialize profiler
    profiler = PerformanceProfiler()
    
    # Generate test data
    n_clones, x, a = profiler.generate_test_data(n_steps=150000)
    
    print(f"\nDataset info:")
    print(f"  Sequence length: {len(x):,}")
    print(f"  States: {n_clones.sum().item():,}")
    print(f"  Actions: {a.max().item() + 1}")
    print(f"  Device: {x.device}")
    
    # Profile initialization
    model = profiler.profile_chmm_initialization(n_clones, x, a)
    
    # Check device alignment
    profiler.check_device_transfers(model, x, a)
    
    # Profile individual components
    log2_lik, mess_fwd = profiler.profile_forward_pass(model, x, a)
    mess_bwd = profiler.profile_backward_pass(model, x, a)
    C = profiler.profile_update_counts(model, x, a, mess_fwd, mess_bwd)
    
    # Profile full iteration
    model_trained, progression = profiler.profile_full_training_iteration(n_clones, x, a)
    
    # Analyze assert overhead
    profiler.analyze_assert_overhead(model, x, a)
    
    # Generate comprehensive report
    report = profiler.generate_report()
    
    print(f"\n🎯 Key findings:")
    print(f"   Total time: {report['total_time']:.1f}s")
    print(f"   GPU efficiency: {report['efficiency']:.1f}%")
    print(f"   Main bottleneck: {report['bottlenecks'][0][0]}")
    
    if report['total_time'] > 180:  # 3 minutes
        print(f"⚠️  Training time ({report['total_time']:.1f}s) exceeds 3 minutes - optimization needed")
    
    return report

if __name__ == "__main__":
    report = main()