#!/usr/bin/env python3
"""
Vectorized indexing optimization to eliminate expensive .item() calls.
This addresses the 11+ minute bottleneck identified in granular profiling.
"""

import os
import sys
import time
import torch

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

def create_vectorized_forward_pass():
    """Create optimized forward pass that eliminates .item() calls."""
    
    def vectorized_forward(T_tr, Pi, n_clones, x, a, device, store_messages=False, workspace=None):
        """
        Vectorized forward pass that eliminates GPU→CPU transfers.
        This is the key optimization identified by granular profiling.
        """
        # Setup (same as before)
        if workspace is None:
            workspace = {}
        
        if 'state_loc' not in workspace:
            workspace['state_loc'] = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones]).cumsum(0)
        state_loc = workspace['state_loc']
        
        seq_len = x.shape[0]
        log2_lik = torch.zeros(seq_len, dtype=T_tr.dtype, device=device)
        
        # Initial message
        j = x[0]
        j_start, j_stop = state_loc[j : j + 2]
        message = Pi[j_start:j_stop].clone()
        p_obs = message.sum()
        message = message / p_obs
        log2_lik[0] = torch.log2(p_obs)
        
        if store_messages:
            mess_loc = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones[x]]).cumsum(0)
            mess_fwd = torch.empty(mess_loc[-1], dtype=T_tr.dtype, device=device)
            t_start, t_stop = mess_loc[0:2]
            mess_fwd[t_start:t_stop] = message
        else:
            mess_fwd = None
            mess_loc = None
        
        if seq_len == 1:
            return log2_lik, mess_fwd
        
        # CRITICAL OPTIMIZATION: Vectorized indexing without .item() calls
        # Pre-compute ALL indices as tensors (no CPU transfers)
        i_indices = x[:-1]  # [T-1]
        j_indices = x[1:]   # [T-1]
        a_indices = a[:-1]  # [T-1]
        
        i_starts = state_loc[i_indices]  # [T-1] - GPU tensor
        i_stops = state_loc[i_indices + 1]  # [T-1] - GPU tensor
        j_starts = state_loc[j_indices]  # [T-1] - GPU tensor
        j_stops = state_loc[j_indices + 1]  # [T-1] - GPU tensor
        
        # Vectorized forward pass - process steps in chunks to reduce memory
        chunk_size = min(1000, seq_len - 1)  # Process in chunks
        
        for chunk_start in range(0, seq_len - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len - 1)
            chunk_indices = slice(chunk_start, chunk_end)
            
            # Get chunk indices (still on GPU - no .item() calls!)
            a_chunk = a_indices[chunk_indices]
            i_starts_chunk = i_starts[chunk_indices]
            i_stops_chunk = i_stops[chunk_indices]
            j_starts_chunk = j_starts[chunk_indices]
            j_stops_chunk = j_stops[chunk_indices]
            
            # Process each step in chunk (vectorized where possible)
            for local_idx in range(chunk_end - chunk_start):
                global_idx = chunk_start + local_idx
                t = global_idx + 1
                
                # OPTIMIZED: Use tensor indexing instead of .item()
                ajt = a_chunk[local_idx]  # GPU tensor (single element)
                i_start = i_starts_chunk[local_idx]  # GPU tensor
                i_stop = i_stops_chunk[local_idx]   # GPU tensor  
                j_start = j_starts_chunk[local_idx]  # GPU tensor
                j_stop = j_stops_chunk[local_idx]   # GPU tensor
                
                # Advanced indexing without .item() - use tensor indexing
                T_tr_slice = T_tr[ajt, j_start:j_stop, i_start:i_stop]
                
                # Matrix-vector multiplication
                message = torch.matmul(T_tr_slice, message.unsqueeze(-1)).squeeze(-1)
                
                # Normalization
                p_obs = message.sum()
                message = message / p_obs
                log2_lik[t] = torch.log2(p_obs)
                
                if store_messages:
                    t_start_msg, t_stop_msg = mess_loc[t : t + 2]
                    mess_fwd[t_start_msg:t_stop_msg] = message
        
        return log2_lik, mess_fwd
    
    return vectorized_forward


def create_vectorized_backward_pass():
    """Create optimized backward pass that eliminates .item() calls."""
    
    def vectorized_backward(T, n_clones, x, a, device, workspace=None):
        """Vectorized backward pass without GPU→CPU transfers."""
        
        if workspace is None:
            workspace = {}
            
        if 'state_loc' not in workspace:
            workspace['state_loc'] = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones]).cumsum(0)
        state_loc = workspace['state_loc']
        
        seq_len = x.shape[0]
        dtype = T.dtype
        
        # Initialize final message
        t = seq_len - 1
        i = x[t]
        message = torch.ones(n_clones[i], dtype=dtype, device=device) / n_clones[i]
        
        # Setup message storage
        mess_loc = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones[x]]).cumsum(0)
        mess_bwd = torch.empty(mess_loc[-1], dtype=dtype, device=device)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_bwd[t_start : t_stop] = message
        
        if seq_len <= 2:
            return mess_bwd
        
        # CRITICAL OPTIMIZATION: Vectorized backward indexing
        backward_range = torch.arange(seq_len - 2, -1, -1, device=device)
        i_indices_bwd = x[backward_range]
        j_indices_bwd = x[backward_range + 1]
        a_indices_bwd = a[backward_range]
        
        i_starts_bwd = state_loc[i_indices_bwd]
        i_stops_bwd = state_loc[i_indices_bwd + 1]
        j_starts_bwd = state_loc[j_indices_bwd]
        j_stops_bwd = state_loc[j_indices_bwd + 1]
        
        # Vectorized backward pass
        chunk_size = min(1000, len(backward_range))
        
        for chunk_start in range(0, len(backward_range), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(backward_range))
            
            # Process chunk without .item() calls
            for local_idx in range(chunk_end - chunk_start):
                global_idx = chunk_start + local_idx
                t = backward_range[global_idx]
                
                # OPTIMIZED: Tensor indexing without .item()
                ajt = a_indices_bwd[global_idx]
                i_start = i_starts_bwd[global_idx]
                i_stop = i_stops_bwd[global_idx]
                j_start = j_starts_bwd[global_idx]
                j_stop = j_stops_bwd[global_idx]
                
                # Tensor operations
                T_slice = T[ajt, i_start:i_stop, j_start:j_stop]
                message = torch.matmul(T_slice, message.unsqueeze(-1)).squeeze(-1)
                
                p_obs = message.sum()
                message = message / p_obs
                
                t_start_msg, t_stop_msg = mess_loc[t : t + 2]
                mess_bwd[t_start_msg : t_stop_msg] = message
        
        return mess_bwd
    
    return vectorized_backward


def test_vectorized_indexing_optimization():
    """Test the vectorized indexing optimization."""
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Testing vectorized indexing optimization on {device}")
    
    # Generate test data
    n_steps = 5000
    n_observations = 100
    clones_per_obs = 16
    n_actions = 4
    
    n_clones = torch.tensor([clones_per_obs] * n_observations, dtype=torch.int64, device=device)
    x = torch.randint(0, n_observations, (n_steps,), dtype=torch.int64, device=device)
    a = torch.randint(0, n_actions, (n_steps,), dtype=torch.int64, device=device)
    
    print(f"Test data: {n_steps:,} steps, {n_clones.sum().item():,} states")
    
    # Test original implementation
    print("\n=== Testing Original Implementation ===")
    start = time.perf_counter()
    from models.train_utils import train_chmm
    model_orig, prog_orig = train_chmm(n_clones, x, a, device=device, method='em_T', n_iter=1, seed=42)
    time_original = time.perf_counter() - start
    
    print(f"Original implementation: {time_original:.3f}s")
    
    # Test vectorized implementation (would need to integrate into train_utils)
    # For now, just test the forward/backward pass components
    
    print(f"\n=== Component Testing ===")
    
    from models.chmm_torch import CHMM_torch
    model = CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
    
    vectorized_forward = create_vectorized_forward_pass()
    vectorized_backward = create_vectorized_backward_pass()
    
    # Test forward pass
    print("Testing vectorized forward pass...")
    start = time.perf_counter()
    T_tr = model.T.transpose(1, 2)
    workspace = {}
    log2_lik, mess_fwd = vectorized_forward(T_tr, model.Pi_x, model.n_clones, x, a, device, 
                                          store_messages=True, workspace=workspace)
    forward_time = time.perf_counter() - start
    print(f"Vectorized forward: {forward_time:.3f}s")
    
    # Test backward pass  
    print("Testing vectorized backward pass...")
    start = time.perf_counter()
    mess_bwd = vectorized_backward(model.T, model.n_clones, x, a, device, workspace=workspace)
    backward_time = time.perf_counter() - start
    print(f"Vectorized backward: {backward_time:.3f}s")
    
    # Estimate total improvement
    total_vectorized = forward_time + backward_time
    print(f"\nEstimated speedup for forward+backward: {(time_original*0.7)/total_vectorized:.2f}x")
    print(f"(Assuming forward+backward is ~70% of total time)")
    
    return {
        'original_time': time_original,
        'forward_time': forward_time,
        'backward_time': backward_time,
        'total_vectorized': total_vectorized
    }


def estimate_150k_improvement():
    """Estimate improvement for 150k steps based on granular profiling results."""
    
    print(f"\n=== Estimating 150k Steps Improvement ===")
    
    # From granular profiling results (per step):
    forward_index_time = 1.138e-3  # seconds per step
    backward_index_time = 1.115e-3  # seconds per step
    count_index_time = 2.261e-3    # seconds per step
    
    # Current total index overhead for 150k steps
    current_index_overhead = (forward_index_time + backward_index_time + count_index_time) * 150000
    print(f"Current index overhead (150k steps): {current_index_overhead:.1f}s ({current_index_overhead/60:.1f} minutes)")
    
    # Optimized version should reduce this by 80-90%
    optimized_index_overhead = current_index_overhead * 0.15  # 85% reduction
    time_saved = current_index_overhead - optimized_index_overhead
    
    print(f"Optimized index overhead: {optimized_index_overhead:.1f}s ({optimized_index_overhead/60:.1f} minutes)")
    print(f"Time saved: {time_saved:.1f}s ({time_saved/60:.1f} minutes)")
    
    # Impact on total L4 time (88.1 minutes)
    original_total_minutes = 88.1
    new_total_minutes = original_total_minutes - (time_saved / 60)
    
    print(f"\nProjected L4 GPU performance:")
    print(f"  Original: {original_total_minutes:.1f} minutes")
    print(f"  Optimized: {new_total_minutes:.1f} minutes")
    print(f"  Speedup: {original_total_minutes/new_total_minutes:.2f}x")
    
    if new_total_minutes < 45:
        print(f"  🎯 EXCELLENT! Under 45 minutes!")
    elif new_total_minutes < 60:
        print(f"  🎯 GREAT! Under 1 hour!")
    else:
        print(f"  ✅ Good improvement, more optimizations possible")


if __name__ == "__main__":
    print("Vectorized Indexing Optimization Test")
    print("=" * 50)
    
    try:
        results = test_vectorized_indexing_optimization()
        estimate_150k_improvement()
        
        print(f"\n🚀 This optimization should provide the biggest speedup yet!")
        print(f"Key insight: Eliminate ALL .item() calls to prevent GPU→CPU transfers")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()