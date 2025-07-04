#!/usr/bin/env python3
"""
Test scaling of CPU vs GPU with different problem sizes
Baum-Welch is O(N^2 * M) where N=num_states, M=sequence_length
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.chmm_torch import CHMM_torch
from env_adapters.room_adapter import RoomAdapter
from agent_adapters.agent_2d import Agent2D

def test_single_iteration(device, n_clones_per_obs, seq_len, room_size=5):
    """Test single EM iteration performance"""
    
    # Create synthetic room
    room = np.random.randint(0, 16, (room_size, room_size))
    
    # Generate trajectory
    env = RoomAdapter(room, seed=42)
    agent = Agent2D(seed=42)
    observations, actions, path = agent.traverse(env, seq_len)
    
    # Move to target device
    x_torch = observations.to(device)
    a_torch = actions.to(device)
    
    # Setup model
    n_emissions = 16  # Fixed for comparison
    n_clones = torch.ones(n_emissions, dtype=torch.int64, device=device) * n_clones_per_obs
    total_states = n_clones.sum().item()
    
    model = CHMM_torch(
        n_clones=n_clones,
        x=x_torch,
        a=a_torch,
        pseudocount=0.01,
        seed=42,
        device=device
    )
    
    # Prepare workspace
    if not hasattr(model, '_workspace'):
        model._workspace = {}
    model._T_transposed = model.T.permute(0, 2, 1).contiguous()
    
    # Time single EM iteration
    start_time = time.time()
    
    from models.train_utils import forward, backward, updateC
    
    log2_lik, mess_fwd = forward(
        model._T_transposed, model.Pi_x, model.n_clones,
        x_torch, a_torch, model.device, store_messages=True, workspace=model._workspace
    )
    
    mess_bwd = backward(
        model.T, model.n_clones, x_torch, a_torch, model.device, workspace=model._workspace
    )
    
    updateC(
        model.C, model.T, model.n_clones, mess_fwd, mess_bwd, 
        x_torch, a_torch, model.device, workspace=model._workspace
    )
    
    model.update_T()
    
    total_time = time.time() - start_time
    
    return total_time, total_states

def test_scaling():
    """Test scaling with different problem sizes"""
    
    devices = []
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    devices.append(torch.device("cpu"))
    
    # Test different numbers of states (N in O(N^2*M))
    n_clones_list = [10, 100, 1000]
    seq_len = 5000  # Fixed sequence length
    
    results = {str(device): {'n_states': [], 'times': []} for device in devices}
    
    print("SCALING TEST: Varying Number of States")
    print("=" * 60)
    print(f"Sequence length: {seq_len}")
    print(f"Testing n_clones_per_obs: {n_clones_list}")
    print()
    
    for n_clones_per_obs in n_clones_list:
        print(f"\nTesting {n_clones_per_obs} clones per obs:")
        
        for device in devices:
            try:
                total_time, total_states = test_single_iteration(
                    device, n_clones_per_obs, seq_len
                )
                
                results[str(device)]['n_states'].append(total_states)
                results[str(device)]['times'].append(total_time)
                
                print(f"  {device}: {total_time:.3f}s ({total_states} states)")
                
            except Exception as e:
                print(f"  {device}: ERROR - {e}")
    
    # Test different sequence lengths (M in O(N^2*M))
    print(f"\n{'='*60}")
    print("SCALING TEST: Varying Sequence Length")
    print("=" * 60)
    
    n_clones_per_obs = 30  # Fixed number of clones
    seq_lengths = [1000, 10000, 100000, 1000000]
    
    seq_results = {str(device): {'seq_lens': [], 'times': []} for device in devices}
    
    print(f"Number of clones per obs: {n_clones_per_obs}")
    print(f"Testing sequence lengths: {seq_lengths}")
    print()
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length {seq_len}:")
        
        for device in devices:
            try:
                total_time, total_states = test_single_iteration(
                    device, n_clones_per_obs, seq_len
                )
                
                seq_results[str(device)]['seq_lens'].append(seq_len)
                seq_results[str(device)]['times'].append(total_time)
                
                steps_per_sec = seq_len / total_time
                print(f"  {device}: {total_time:.3f}s ({steps_per_sec:.0f} steps/sec)")
                
            except Exception as e:
                print(f"  {device}: ERROR - {e}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print("=" * 60)
    
    # Find crossover point for states
    print("\n1. States scaling (N^2 complexity):")
    cpu_key = 'cpu'
    mps_key = None
    for key in results.keys():
        if 'mps' in key:
            mps_key = key
            break
    
    if mps_key and len(results[cpu_key]['times']) > 0 and len(results[mps_key]['times']) > 0:
        cpu_times = results[cpu_key]['times']
        mps_times = results[mps_key]['times']
        n_states = results[cpu_key]['n_states']
        
        for i, (cpu_t, mps_t, states) in enumerate(zip(cpu_times, mps_times, n_states)):
            ratio = mps_t / cpu_t
            print(f"  {states:4d} states: CPU {cpu_t:.3f}s, MPS {mps_t:.3f}s (MPS {ratio:.1f}x slower)")
            
            if ratio < 2.0:  # GPU becomes competitive when <2x slower
                print(f"  >>> GPU becomes competitive around {states} states")
                break
    
    # Find crossover point for sequence length
    print("\n2. Sequence length scaling (M complexity):")
    if mps_key and len(seq_results[cpu_key]['times']) > 0 and len(seq_results[mps_key]['times']) > 0:
        cpu_times = seq_results[cpu_key]['times']
        mps_times = seq_results[mps_key]['times']
        seq_lens = seq_results[cpu_key]['seq_lens']
        
        for i, (cpu_t, mps_t, seq_len) in enumerate(zip(cpu_times, mps_times, seq_lens)):
            ratio = mps_t / cpu_t
            print(f"  {seq_len:5d} steps: CPU {cpu_t:.3f}s, MPS {mps_t:.3f}s (MPS {ratio:.1f}x slower)")
            
            if ratio < 2.0:
                print(f"  >>> GPU becomes competitive around {seq_len} steps")
                break
    
    # Recommendations
    print(f"\n3. RECOMMENDATIONS:")
    print("  For small problems (5x5 room, 30 clones/obs, <20k steps):")
    print("    - Use CPU (10-20x faster)")
    print("    - GPU overhead dominates")
    print()
    print("  GPU likely beneficial for:")
    print("    - Large rooms (20x20, 50x50)")
    print("    - Many clones (>100 per obs)")
    print("    - Very long sequences (>50k steps)")
    print()
    print("  Current config should use CPU for 5x5 rooms!")

if __name__ == "__main__":
    test_scaling()