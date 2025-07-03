#!/usr/bin/env python3
"""
Ultra-granular Baum-Welch algorithm profiler for MPS/GPU optimization.
This dissects every single operation in the algorithm to find optimization opportunities.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

class GranularBaumWelchProfiler:
    """Ultra-granular profiler for every operation in Baum-Welch algorithm."""
    
    def __init__(self, device=None):
        self.device = device or self._detect_device()
        self.timings = {}
        self.operation_counts = {}
        self.memory_usage = {}
        
        print(f"Granular Baum-Welch Profiler")
        print(f"Device: {self.device}")
        if self.device.type == 'mps':
            print("Optimizing for Apple Silicon MPS")
        elif self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def _detect_device(self):
        """Detect optimal device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _time_operation(self, name: str, func, *args, **kwargs):
        """Time an operation with proper synchronization."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_memory = 0
        if self.device.type == 'cuda':
            start_memory = torch.cuda.memory_allocated()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        
        if self.device.type == 'cuda':
            end_memory = torch.cuda.memory_allocated()
            self.memory_usage[name] = (end_memory - start_memory) / 1024**2  # MB
        
        self.timings[name] = elapsed
        return result, elapsed
    
    def generate_test_data(self, n_steps=5000, n_observations=100, clones_per_obs=16, n_actions=4):
        """Generate test data for profiling."""
        print(f"\nGenerating test data: {n_steps} steps, {n_observations} obs, {clones_per_obs} clones/obs")
        
        n_clones = torch.tensor([clones_per_obs] * n_observations, dtype=torch.int64, device=self.device)
        x = torch.randint(0, n_observations, (n_steps,), dtype=torch.int64, device=self.device)
        a = torch.randint(0, n_actions, (n_steps,), dtype=torch.int64, device=self.device)
        
        total_states = n_clones.sum().item()
        print(f"Total states: {total_states:,}")
        
        return n_clones, x, a, total_states
    
    def profile_data_preparation(self, n_clones, x, a):
        """Profile data preparation operations."""
        print(f"\n=== Profiling Data Preparation ===")
        
        # State location computation
        def compute_state_loc():
            return torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=self.device), n_clones]).cumsum(0)
        
        state_loc, _ = self._time_operation("state_loc_computation", compute_state_loc)
        
        # Message location computation
        def compute_mess_loc():
            return torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=self.device), n_clones[x]]).cumsum(0)
        
        mess_loc, _ = self._time_operation("mess_loc_computation", compute_mess_loc)
        
        # Index precomputation
        def precompute_indices():
            seq_len = x.shape[0]
            if seq_len > 1:
                i_indices = x[:-1]
                j_indices = x[1:]
                a_indices = a[:-1]
                
                i_starts = state_loc[i_indices]
                i_stops = state_loc[i_indices + 1]
                j_starts = state_loc[j_indices]
                j_stops = state_loc[j_indices + 1]
                
                return i_indices, j_indices, a_indices, i_starts, i_stops, j_starts, j_stops
            return None
        
        indices, _ = self._time_operation("index_precomputation", precompute_indices)
        
        return state_loc, mess_loc, indices
    
    def profile_model_initialization(self, n_clones, x, a):
        """Profile CHMM model initialization."""
        print(f"\n=== Profiling Model Initialization ===")
        
        from models.chmm_torch import CHMM_torch
        
        def init_model():
            return CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
        
        model, _ = self._time_operation("model_initialization", init_model)
        
        # Break down initialization components
        n_states = n_clones.sum().item()
        n_actions = a.max().item() + 1
        
        def create_transition_matrix():
            return torch.randn(n_actions, n_states, n_states, device=self.device, dtype=torch.float32)
        
        _, _ = self._time_operation("transition_matrix_creation", create_transition_matrix)
        
        def create_initial_distribution():
            return torch.randn(n_states, device=self.device, dtype=torch.float32)
        
        _, _ = self._time_operation("initial_distribution_creation", create_initial_distribution)
        
        return model
    
    def profile_forward_pass_granular(self, model, x, a, state_loc, mess_loc, indices):
        """Ultra-granular profiling of forward pass."""
        print(f"\n=== Granular Forward Pass Profiling ===")
        
        T_tr = model.T.transpose(1, 2)
        Pi = model.Pi_x
        n_clones = model.n_clones
        
        # Initialize workspace
        workspace = {}
        log2_lik = torch.zeros(len(x), dtype=T_tr.dtype, device=self.device)
        
        # Profile initial message setup
        def setup_initial_message():
            t = 0
            j = x[t]
            j_start, j_stop = state_loc[j : j + 2]
            message = Pi[j_start:j_stop].clone()
            p_obs = message.sum()
            message = message / p_obs
            log2_lik[0] = torch.log2(p_obs)
            return message
        
        message, _ = self._time_operation("initial_message_setup", setup_initial_message)
        
        # Profile message storage initialization
        def setup_message_storage():
            mess_fwd = torch.empty(mess_loc[-1], dtype=T_tr.dtype, device=self.device)
            t_start, t_stop = mess_loc[0:2]
            mess_fwd[t_start:t_stop] = message
            return mess_fwd
        
        mess_fwd, _ = self._time_operation("message_storage_setup", setup_message_storage)
        
        # Profile individual forward step components
        seq_len = x.shape[0]
        if seq_len > 1 and indices is not None:
            i_indices, j_indices, a_indices, i_starts, i_stops, j_starts, j_stops = indices
            
            # Profile just the core matrix operations
            step_timings = {
                'index_access': [],
                'tensor_slicing': [],
                'matrix_vector_mult': [],
                'normalization': [],
                'message_storage': []
            }
            
            # Profile first 100 steps in detail
            n_profile_steps = min(100, seq_len - 1)
            
            for step_idx in range(n_profile_steps):
                t = step_idx + 1
                
                # Index access timing
                start = time.perf_counter()
                ajt = a_indices[step_idx].item()
                i_start, i_stop = i_starts[step_idx].item(), i_stops[step_idx].item()
                j_start, j_stop = j_starts[step_idx].item(), j_stops[step_idx].item()
                step_timings['index_access'].append(time.perf_counter() - start)
                
                # Tensor slicing timing
                start = time.perf_counter()
                T_tr_slice = T_tr[ajt, j_start:j_stop, i_start:i_stop]
                step_timings['tensor_slicing'].append(time.perf_counter() - start)
                
                # Matrix-vector multiplication timing
                start = time.perf_counter()
                message = torch.matmul(T_tr_slice, message.unsqueeze(-1)).squeeze(-1)
                step_timings['matrix_vector_mult'].append(time.perf_counter() - start)
                
                # Normalization timing
                start = time.perf_counter()
                p_obs = message.sum()
                message = message / p_obs
                log2_lik[t] = torch.log2(p_obs)
                step_timings['normalization'].append(time.perf_counter() - start)
                
                # Message storage timing
                start = time.perf_counter()
                t_start, t_stop = mess_loc[t : t + 2]
                mess_fwd[t_start:t_stop] = message
                step_timings['message_storage'].append(time.perf_counter() - start)
            
            # Store average timings for each operation
            for op_name, times in step_timings.items():
                avg_time = np.mean(times)
                self.timings[f"forward_step_{op_name}"] = avg_time
                self.operation_counts[f"forward_step_{op_name}"] = len(times)
                print(f"  {op_name}: {avg_time*1000:.3f}ms avg per step")
        
        return log2_lik, mess_fwd
    
    def profile_backward_pass_granular(self, model, x, a, state_loc, mess_loc):
        """Ultra-granular profiling of backward pass."""
        print(f"\n=== Granular Backward Pass Profiling ===")
        
        T = model.T
        n_clones = model.n_clones
        dtype = T.dtype
        seq_len = x.shape[0]
        
        # Profile backward initialization
        def setup_backward_initial():
            t = seq_len - 1
            i = x[t]
            message = torch.ones(n_clones[i], dtype=dtype, device=self.device) / n_clones[i]
            return message
        
        message, _ = self._time_operation("backward_initial_setup", setup_backward_initial)
        
        # Profile backward storage setup
        def setup_backward_storage():
            mess_bwd = torch.empty(mess_loc[-1], dtype=dtype, device=self.device)
            t = seq_len - 1
            t_start, t_stop = mess_loc[t : t + 2]
            mess_bwd[t_start : t_stop] = message
            return mess_bwd
        
        mess_bwd, _ = self._time_operation("backward_storage_setup", setup_backward_storage)
        
        # Profile backward index computation
        def compute_backward_indices():
            if seq_len > 2:
                backward_range = torch.arange(seq_len - 2, -1, -1, device=self.device)
                i_indices_bwd = x[backward_range]
                j_indices_bwd = x[backward_range + 1]
                a_indices_bwd = a[backward_range]
                
                i_starts_bwd = state_loc[i_indices_bwd]
                i_stops_bwd = state_loc[i_indices_bwd + 1]
                j_starts_bwd = state_loc[j_indices_bwd]
                j_stops_bwd = state_loc[j_indices_bwd + 1]
                
                return a_indices_bwd, i_starts_bwd, i_stops_bwd, j_starts_bwd, j_stops_bwd
            return None, None, None, None, None
        
        bwd_indices, _ = self._time_operation("backward_index_computation", compute_backward_indices)
        
        # Profile individual backward step components
        if seq_len > 2 and bwd_indices[0] is not None:
            a_indices_bwd, i_starts_bwd, i_stops_bwd, j_starts_bwd, j_stops_bwd = bwd_indices
            
            step_timings = {
                'index_access': [],
                'tensor_slicing': [],
                'matrix_vector_mult': [],
                'normalization': [],
                'message_storage': []
            }
            
            n_profile_steps = min(100, seq_len - 2)
            
            for idx in range(n_profile_steps):
                t = seq_len - 2 - idx
                
                # Index access timing
                start = time.perf_counter()
                ajt = a_indices_bwd[idx].item()
                i_start, i_stop = i_starts_bwd[idx].item(), i_stops_bwd[idx].item()
                j_start, j_stop = j_starts_bwd[idx].item(), j_stops_bwd[idx].item()
                step_timings['index_access'].append(time.perf_counter() - start)
                
                # Tensor slicing timing
                start = time.perf_counter()
                T_slice = T[ajt, i_start:i_stop, j_start:j_stop]
                step_timings['tensor_slicing'].append(time.perf_counter() - start)
                
                # Matrix-vector multiplication timing
                start = time.perf_counter()
                message = torch.matmul(T_slice, message.unsqueeze(-1)).squeeze(-1)
                step_timings['matrix_vector_mult'].append(time.perf_counter() - start)
                
                # Normalization timing
                start = time.perf_counter()
                p_obs = message.sum()
                message = message / p_obs
                step_timings['normalization'].append(time.perf_counter() - start)
                
                # Message storage timing
                start = time.perf_counter()
                t_start, t_stop = mess_loc[t : t + 2]
                mess_bwd[t_start : t_stop] = message
                step_timings['message_storage'].append(time.perf_counter() - start)
            
            # Store average timings
            for op_name, times in step_timings.items():
                avg_time = np.mean(times)
                self.timings[f"backward_step_{op_name}"] = avg_time
                self.operation_counts[f"backward_step_{op_name}"] = len(times)
                print(f"  {op_name}: {avg_time*1000:.3f}ms avg per step")
        
        return mess_bwd
    
    def profile_count_update_granular(self, model, x, a, mess_fwd, mess_bwd, state_loc, mess_loc):
        """Ultra-granular profiling of count update (E-step)."""
        print(f"\n=== Granular Count Update Profiling ===")
        
        T = model.T
        n_clones = model.n_clones
        timesteps = len(x)
        
        # Profile count matrix initialization
        def init_count_matrix():
            return torch.zeros_like(T)
        
        C, _ = self._time_operation("count_matrix_initialization", init_count_matrix)
        
        # Profile index computation for updates
        def compute_update_indices():
            if timesteps > 1:
                t_range = torch.arange(1, timesteps, device=self.device)
                i_indices_upd = x[:-1]
                j_indices_upd = x[1:]
                a_indices_upd = a[:-1]
                
                tm1_starts = mess_loc[t_range - 1]
                tm1_stops = mess_loc[t_range]
                t_starts = mess_loc[t_range]
                t_stops = mess_loc[t_range + 1]
                i_starts_upd = state_loc[i_indices_upd]
                i_stops_upd = state_loc[i_indices_upd + 1]
                j_starts_upd = state_loc[j_indices_upd]
                j_stops_upd = state_loc[j_indices_upd + 1]
                
                return (a_indices_upd, tm1_starts, tm1_stops, t_starts, t_stops,
                       i_starts_upd, i_stops_upd, j_starts_upd, j_stops_upd)
            return None
        
        update_indices, _ = self._time_operation("update_index_computation", compute_update_indices)
        
        # Profile individual count update steps
        if timesteps > 1 and update_indices is not None:
            (a_indices_upd, tm1_starts, tm1_stops, t_starts, t_stops,
             i_starts_upd, i_stops_upd, j_starts_upd, j_stops_upd) = update_indices
            
            step_timings = {
                'index_access': [],
                'message_extraction': [],
                'tensor_slicing': [],
                'outer_product': [],
                'normalization': [],
                'accumulation': []
            }
            
            n_profile_steps = min(100, timesteps - 1)
            
            for idx in range(n_profile_steps):
                # Index access timing
                start = time.perf_counter()
                ajt = a_indices_upd[idx].item()
                tm1_start, tm1_stop = tm1_starts[idx].item(), tm1_stops[idx].item()
                t_start, t_stop = t_starts[idx].item(), t_stops[idx].item()
                i_start, i_stop = i_starts_upd[idx].item(), i_stops_upd[idx].item()
                j_start, j_stop = j_starts_upd[idx].item(), j_stops_upd[idx].item()
                step_timings['index_access'].append(time.perf_counter() - start)
                
                # Message extraction timing
                start = time.perf_counter()
                alpha = mess_fwd[tm1_start:tm1_stop]
                beta = mess_bwd[t_start:t_stop]
                step_timings['message_extraction'].append(time.perf_counter() - start)
                
                # Tensor slicing timing
                start = time.perf_counter()
                T_slice = T[ajt, i_start:i_stop, j_start:j_stop]
                step_timings['tensor_slicing'].append(time.perf_counter() - start)
                
                # Outer product timing
                start = time.perf_counter()
                q = torch.outer(alpha, beta) * T_slice
                step_timings['outer_product'].append(time.perf_counter() - start)
                
                # Normalization timing
                start = time.perf_counter()
                norm = q.sum()
                if norm > 0:
                    q /= norm
                step_timings['normalization'].append(time.perf_counter() - start)
                
                # Accumulation timing
                start = time.perf_counter()
                if norm > 0:
                    C[ajt, i_start:i_stop, j_start:j_stop] += q
                step_timings['accumulation'].append(time.perf_counter() - start)
            
            # Store average timings
            for op_name, times in step_timings.items():
                avg_time = np.mean(times)
                self.timings[f"count_update_{op_name}"] = avg_time
                self.operation_counts[f"count_update_{op_name}"] = len(times)
                print(f"  {op_name}: {avg_time*1000:.3f}ms avg per step")
        
        return C
    
    def profile_memory_operations(self, n_states, n_actions, device):
        """Profile basic memory operations on the device."""
        print(f"\n=== Memory Operations Profiling ===")
        
        # Profile tensor creation
        def create_large_tensor():
            return torch.randn(n_actions, n_states, n_states, device=device, dtype=torch.float32)
        
        _, _ = self._time_operation("large_tensor_creation", create_large_tensor)
        
        # Profile tensor indexing
        T = torch.randn(n_actions, n_states, n_states, device=device)
        
        def tensor_indexing():
            for i in range(100):
                action = i % n_actions
                start = (i * 10) % (n_states - 100)
                end = start + 100
                _ = T[action, start:end, start:end]
        
        _, _ = self._time_operation("tensor_indexing_100ops", tensor_indexing)
        
        # Profile memory copies
        def memory_copy():
            src = torch.randn(1000, 1000, device=device)
            dst = src.clone()
            return dst
        
        _, _ = self._time_operation("memory_copy_1M_elements", memory_copy)
        
        # Profile matrix operations
        A = torch.randn(1000, 1000, device=device)
        B = torch.randn(1000, 1000, device=device)
        
        def matrix_multiply():
            return torch.matmul(A, B)
        
        _, _ = self._time_operation("matrix_multiply_1000x1000", matrix_multiply)
        
        # Profile vector operations
        v1 = torch.randn(1000, device=device)
        v2 = torch.randn(1000, device=device)
        
        def outer_product():
            return torch.outer(v1, v2)
        
        _, _ = self._time_operation("outer_product_1000x1000", outer_product)
    
    def run_complete_profiling(self, n_steps=5000, n_observations=100, clones_per_obs=16, n_actions=4):
        """Run complete granular profiling of Baum-Welch algorithm."""
        print("=" * 80)
        print("GRANULAR BAUM-WELCH ALGORITHM PROFILER")
        print("=" * 80)
        
        # Generate test data
        n_clones, x, a, total_states = self.generate_test_data(n_steps, n_observations, clones_per_obs, n_actions)
        
        # Profile data preparation
        state_loc, mess_loc, indices = self.profile_data_preparation(n_clones, x, a)
        
        # Profile model initialization
        model = self.profile_model_initialization(n_clones, x, a)
        
        # Profile memory operations
        self.profile_memory_operations(total_states, n_actions, self.device)
        
        # Profile forward pass
        log2_lik, mess_fwd = self.profile_forward_pass_granular(model, x, a, state_loc, mess_loc, indices)
        
        # Profile backward pass
        mess_bwd = self.profile_backward_pass_granular(model, x, a, state_loc, mess_loc)
        
        # Profile count update
        C = self.profile_count_update_granular(model, x, a, mess_fwd, mess_bwd, state_loc, mess_loc)
        
        return self.generate_optimization_report()
    
    def generate_optimization_report(self):
        """Generate detailed optimization recommendations."""
        print(f"\n" + "=" * 80)
        print("GRANULAR OPTIMIZATION ANALYSIS")
        print("=" * 80)
        
        # Sort operations by time
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 20 Most Expensive Operations:")
        total_time = sum(self.timings.values())
        
        for i, (name, time_val) in enumerate(sorted_timings[:20]):
            pct = (time_val / total_time) * 100 if total_time > 0 else 0
            ops_count = self.operation_counts.get(name, 1)
            
            if 'step_' in name:
                # For per-step operations, show total time if scaled up
                total_if_scaled = time_val * 5000  # Scale to full sequence
                print(f"  {i+1:2d}. {name:<35}: {time_val*1000:8.3f}ms avg ({pct:5.1f}%) -> {total_if_scaled:.3f}s total")
            else:
                print(f"  {i+1:2d}. {name:<35}: {time_val*1000:8.3f}ms ({pct:5.1f}%)")
        
        # Analyze bottlenecks by category
        categories = {
            'data_prep': ['state_loc', 'mess_loc', 'index_'],
            'initialization': ['model_', 'initial_', 'creation'],
            'forward_ops': ['forward_step_'],
            'backward_ops': ['backward_step_'],
            'count_update_ops': ['count_update_'],
            'memory_ops': ['tensor_', 'memory_', 'matrix_', 'outer_']
        }
        
        print(f"\nBottleneck Analysis by Category:")
        category_times = {}
        
        for category, keywords in categories.items():
            category_time = 0
            for name, time_val in self.timings.items():
                if any(kw in name for kw in keywords):
                    if 'step_' in name:
                        # Scale up per-step operations
                        category_time += time_val * 5000
                    else:
                        category_time += time_val
            
            category_times[category] = category_time
            pct = (category_time / sum(category_times.values())) * 100 if sum(category_times.values()) > 0 else 0
            print(f"  {category:<20}: {category_time:8.3f}s ({pct:5.1f}%)")
        
        # Generate specific optimization recommendations
        print(f"\nSpecific Optimization Recommendations:")
        
        recommendations = []
        
        # Check forward/backward step performance
        fwd_matmul = self.timings.get('forward_step_matrix_vector_mult', 0)
        bwd_matmul = self.timings.get('backward_step_matrix_vector_mult', 0)
        if fwd_matmul > 0 or bwd_matmul > 0:
            total_matmul_time = (fwd_matmul + bwd_matmul) * 5000
            recommendations.append(f"1. Matrix-vector operations: {total_matmul_time:.3f}s total")
            recommendations.append(f"   -> Consider batch matrix operations or different BLAS routines")
        
        # Check outer product performance
        outer_prod_time = self.timings.get('count_update_outer_product', 0) * 5000
        if outer_prod_time > 0.5:
            recommendations.append(f"2. Outer product operations: {outer_prod_time:.3f}s total")
            recommendations.append(f"   -> Consider einsum or optimized tensor contractions")
        
        # Check memory operations
        memory_time = sum(time for name, time in self.timings.items() if any(kw in name for kw in ['tensor_', 'memory_']))
        if memory_time > 0.1:
            recommendations.append(f"3. Memory operations: {memory_time:.3f}s total")
            recommendations.append(f"   -> Pre-allocate workspace tensors, optimize memory access patterns")
        
        # Check indexing overhead
        index_times = [self.timings.get(f'{stage}_step_index_access', 0) for stage in ['forward', 'backward', 'count_update']]
        total_index_time = sum(index_times) * 5000
        if total_index_time > 0.2:
            recommendations.append(f"4. Index access overhead: {total_index_time:.3f}s total")
            recommendations.append(f"   -> Vectorize index computations, reduce .item() calls")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        # Final summary
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"  Total profiled time: {sum(category_times.values()):.3f}s")
        print(f"  Biggest bottleneck: {max(category_times.items(), key=lambda x: x[1])[0]}")
        print(f"  Device: {self.device}")
        
        if self.device.type == 'cuda' and self.memory_usage:
            total_memory = sum(self.memory_usage.values())
            print(f"  Peak memory usage: {total_memory:.1f}MB")
        
        return {
            'timings': self.timings,
            'category_times': category_times,
            'recommendations': recommendations,
            'total_time': sum(category_times.values())
        }

def main():
    """Main profiling function."""
    print("Starting Ultra-Granular Baum-Welch Profiling...")
    
    profiler = GranularBaumWelchProfiler()
    
    # Run profiling with different sizes to understand scaling
    test_sizes = [
        (1000, 50, 16, 4),   # Small test
        (5000, 100, 16, 4),  # Medium test
        (10000, 200, 16, 4), # Large test
    ]
    
    all_results = {}
    
    for n_steps, n_obs, clones_per_obs, n_actions in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing: {n_steps} steps, {n_obs} observations")
        print(f"{'='*60}")
        
        try:
            results = profiler.run_complete_profiling(n_steps, n_obs, clones_per_obs, n_actions)
            all_results[f"{n_steps}_{n_obs}"] = results
        except Exception as e:
            print(f"Error in test {n_steps}_{n_obs}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results across different sizes
    print(f"\n{'='*80}")
    print("SCALING ANALYSIS")
    print(f"{'='*80}")
    
    for test_name, results in all_results.items():
        print(f"\nTest {test_name}:")
        print(f"  Total time: {results['total_time']:.3f}s")
        for category, time_val in results['category_times'].items():
            print(f"  {category}: {time_val:.3f}s")

if __name__ == "__main__":
    main()