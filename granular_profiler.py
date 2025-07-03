#!/usr/bin/env python3
"""
Granular profiler to find the REAL bottlenecks in CHMM training.
Since device transfers aren't the issue, let's find what is.
"""

import os
import sys
import time
import torch
import cProfile
import pstats
from typing import Dict

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

class GranularProfiler:
    """Profile individual operations to find real bottlenecks."""
    
    def __init__(self):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.timings = {}
        
    def time_block(self, name: str):
        """Context manager for timing code blocks."""
        class TimerContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                
            def __enter__(self):
                if self.profiler.device.type in ['cuda', 'mps']:
                    torch.cuda.synchronize() if self.profiler.device.type == 'cuda' else None
                self.start = time.perf_counter()
                return self
                
            def __exit__(self, *args):
                if self.profiler.device.type in ['cuda', 'mps']:
                    torch.cuda.synchronize() if self.profiler.device.type == 'cuda' else None
                elapsed = time.perf_counter() - self.start
                self.profiler.timings[self.name] = elapsed
                
        return TimerContext(self, name)
        
    def profile_matrix_operations(self, n_steps=1000):
        """Profile core matrix operations."""
        print(f"=== Profiling Matrix Operations ({n_steps} steps) ===")
        
        # Create realistic test data
        n_states = 6400  # 400 observations × 16 clones each
        n_actions = 4
        
        # Create matrices similar to CHMM
        T = torch.randn(n_actions, n_states, n_states, device=self.device)
        message = torch.randn(80, device=self.device)  # Average clone size
        
        # Test torch.mv (matrix-vector multiplication)
        with self.time_block("torch_mv"):
            for _ in range(n_steps):
                result = torch.mv(T[0, :80, :80], message)
        
        # Test torch.matmul alternative
        with self.time_block("torch_matmul"):
            for _ in range(n_steps):
                result = torch.matmul(T[0, :80, :80], message)
        
        # Test indexing overhead
        with self.time_block("tensor_indexing"):
            for i in range(n_steps):
                idx = i % n_actions
                slice_tensor = T[idx, :80, :80]
        
        # Test tensor creation overhead
        with self.time_block("tensor_creation"):
            for _ in range(n_steps):
                temp = torch.zeros(80, device=self.device)
        
        print(f"torch.mv: {self.timings['torch_mv']:.3f}s ({self.timings['torch_mv']/n_steps*1000:.2f}ms per op)")
        print(f"torch.matmul: {self.timings['torch_matmul']:.3f}s ({self.timings['torch_matmul']/n_steps*1000:.2f}ms per op)")
        print(f"indexing: {self.timings['tensor_indexing']:.3f}s ({self.timings['tensor_indexing']/n_steps*1000:.2f}ms per op)")
        print(f"creation: {self.timings['tensor_creation']:.3f}s ({self.timings['tensor_creation']/n_steps*1000:.2f}ms per op)")
        
    def profile_forward_pass_components(self):
        """Profile individual components of forward pass."""
        print(f"\n=== Profiling Forward Pass Components ===")
        
        # Setup realistic data
        n_clones = torch.tensor([16] * 400, dtype=torch.int64, device=self.device)
        x = torch.randint(0, 400, (1000,), dtype=torch.int64, device=self.device)
        a = torch.randint(0, 4, (1000,), dtype=torch.int64, device=self.device)
        
        from models.chmm_torch import CHMM_torch
        model = CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
        
        T_tr = model.T.transpose(1, 2)
        Pi = model.Pi_x
        
        # Profile workspace creation
        workspace = {}
        with self.time_block("workspace_setup"):
            state_loc = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=self.device), n_clones]).cumsum(0)
            log2_lik = torch.zeros(len(x), dtype=T_tr.dtype, device=self.device)
            
        # Profile index pre-computation
        with self.time_block("index_precompute"):
            i_indices = x[:-1]
            j_indices = x[1:]
            a_indices = a[:-1]
            i_starts = state_loc[i_indices]
            i_stops = state_loc[i_indices + 1]
            j_starts = state_loc[j_indices]
            j_stops = state_loc[j_indices + 1]
        
        # Profile the actual forward loop (first 100 steps)
        message = Pi[state_loc[x[0]]:state_loc[x[0]+1]].clone()
        
        with self.time_block("forward_loop_100"):
            for t in range(1, min(101, len(x))):
                ajt = a_indices[t-1].item()
                i_start, i_stop = i_starts[t-1].item(), i_stops[t-1].item()
                j_start, j_stop = j_starts[t-1].item(), j_stops[t-1].item()
                
                T_tr_slice = T_tr[ajt, j_start:j_stop, i_start:i_stop]
                message = torch.mv(T_tr_slice, message)
                p_obs = message.sum()
                message = message / p_obs
                log2_lik[t] = torch.log2(p_obs)
        
        # Profile Python loop overhead
        with self.time_block("python_loop_overhead"):
            for t in range(1, min(101, len(x))):
                ajt = a_indices[t-1].item()
                i_start, i_stop = i_starts[t-1].item(), i_stops[t-1].item()
                j_start, j_stop = j_starts[t-1].item(), j_stops[t-1].item()
                # No actual computation, just Python overhead
        
        print(f"Workspace setup: {self.timings['workspace_setup']:.3f}s")
        print(f"Index precompute: {self.timings['index_precompute']:.3f}s")
        print(f"Forward loop (100 steps): {self.timings['forward_loop_100']:.3f}s")
        print(f"Python overhead (100 steps): {self.timings['python_loop_overhead']:.3f}s")
        
        # Calculate per-step costs
        actual_compute = self.timings['forward_loop_100'] - self.timings['python_loop_overhead']
        print(f"Pure computation (100 steps): {actual_compute:.3f}s")
        print(f"Per-step cost: {actual_compute/100*1000:.2f}ms")
        print(f"Estimated 150k steps: {actual_compute/100*150000:.1f}s ({actual_compute/100*150000/60:.1f} minutes)")
        
    def profile_python_vs_vectorized(self):
        """Compare Python loops vs vectorized operations."""
        print(f"\n=== Python vs Vectorized Operations ===")
        
        n_steps = 1000
        x = torch.randint(0, 400, (n_steps,), dtype=torch.int64, device=self.device)
        a = torch.randint(0, 4, (n_steps,), dtype=torch.int64, device=self.device)
        n_clones = torch.tensor([16] * 400, dtype=torch.int64, device=self.device)
        state_loc = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=self.device), n_clones]).cumsum(0)
        
        # Python loop approach (current)
        with self.time_block("python_loop"):
            for t in range(1, n_steps):
                i, j = x[t-1], x[t]
                i_start, i_stop = state_loc[i], state_loc[i+1]
                j_start, j_stop = state_loc[j], state_loc[j+1]
                # Simulate some work
                _ = i_start + i_stop + j_start + j_stop
        
        # Vectorized approach 
        with self.time_block("vectorized"):
            i_indices = x[:-1]
            j_indices = x[1:]
            i_starts = state_loc[i_indices]
            i_stops = state_loc[i_indices + 1]
            j_starts = state_loc[j_indices]
            j_stops = state_loc[j_indices + 1]
            # Vectorized computation
            _ = i_starts + i_stops + j_starts + j_stops
        
        speedup = self.timings['python_loop'] / self.timings['vectorized']
        print(f"Python loop: {self.timings['python_loop']:.3f}s")
        print(f"Vectorized: {self.timings['vectorized']:.3f}s")
        print(f"Vectorization speedup: {speedup:.1f}x")
        
    def profile_memory_access_patterns(self):
        """Profile memory access patterns."""
        print(f"\n=== Memory Access Pattern Analysis ===")
        
        n_states = 6400
        n_actions = 4
        
        # Create large transition matrix
        T = torch.randn(n_actions, n_states, n_states, device=self.device)
        
        # Sequential access (cache-friendly)
        with self.time_block("sequential_access"):
            for i in range(100):
                for j in range(100):
                    _ = T[0, i, j]
        
        # Random access (cache-unfriendly)
        indices = torch.randint(0, n_states, (10000,), device=self.device)
        with self.time_block("random_access"):
            for i in range(100):
                idx = indices[i].item()
                _ = T[0, idx, :100]
        
        print(f"Sequential access: {self.timings['sequential_access']:.3f}s")
        print(f"Random access: {self.timings['random_access']:.3f}s")
        
    def run_cprofile_analysis(self):
        """Run detailed cProfile analysis."""
        print(f"\n=== Python cProfile Analysis ===")
        
        # Setup
        n_clones = torch.tensor([16] * 100, dtype=torch.int64, device=self.device)  # Smaller for profiling
        x = torch.randint(0, 100, (1000,), dtype=torch.int64, device=self.device)
        a = torch.randint(0, 4, (1000,), dtype=torch.int64, device=self.device)
        
        def train_small():
            from models.train_utils import train_chmm
            return train_chmm(n_clones, x, a, device=self.device, method='em_T', n_iter=1)
        
        # Run with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        model, progression = train_small()
        profiler.disable()
        
        # Analyze results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        print("Top 10 most time-consuming functions:")
        stats.print_stats(10)
        
    def generate_optimization_report(self):
        """Generate comprehensive optimization recommendations."""
        print(f"\n" + "="*60)
        print("OPTIMIZATION ANALYSIS REPORT")
        print("="*60)
        
        # Sort timings by duration
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nBottlenecks (by time):")
        total_time = sum(self.timings.values())
        for name, time_val in sorted_timings[:10]:
            pct = (time_val / total_time) * 100 if total_time > 0 else 0
            print(f"  {name}: {time_val:.3f}s ({pct:.1f}%)")
        
        print(f"\nKey Findings:")
        
        # Analyze matrix operation efficiency
        if 'torch_mv' in self.timings and 'torch_matmul' in self.timings:
            mv_vs_matmul = self.timings['torch_matmul'] / self.timings['torch_mv']
            print(f"  torch.mv vs torch.matmul: {mv_vs_matmul:.1f}x difference")
        
        # Analyze Python overhead
        if 'forward_loop_100' in self.timings and 'python_loop_overhead' in self.timings:
            python_overhead_pct = (self.timings['python_loop_overhead'] / self.timings['forward_loop_100']) * 100
            print(f"  Python loop overhead: {python_overhead_pct:.1f}% of forward pass")
        
        # Analyze vectorization benefit
        if 'python_loop' in self.timings and 'vectorized' in self.timings:
            vec_speedup = self.timings['python_loop'] / self.timings['vectorized']
            print(f"  Vectorization speedup: {vec_speedup:.1f}x")
        
        print(f"\nRecommendations for A100:")
        print(f"  1. Focus on algorithmic optimizations (O(n²) -> O(n log n))")
        print(f"  2. Implement chunked processing for large sequences")
        print(f"  3. Use tensor parallelization for matrix operations")
        print(f"  4. Consider mixed precision (FP16) for memory bandwidth")
        print(f"  5. Implement sequence batching if possible")

def main():
    """Main profiling function."""
    print("Granular CHMM Performance Profiler")
    print("Finding the REAL bottlenecks...")
    print("="*50)
    
    profiler = GranularProfiler()
    
    # Run all profiling tests
    profiler.profile_matrix_operations(1000)
    profiler.profile_forward_pass_components()
    profiler.profile_python_vs_vectorized()
    profiler.profile_memory_access_patterns()
    
    # Generate final report
    profiler.generate_optimization_report()
    
    print(f"\n🎯 This analysis will help identify the real A100 bottlenecks!")

if __name__ == "__main__":
    main()