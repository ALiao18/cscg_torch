#!/usr/bin/env python3
"""
Compare CPU vs GPU performance for CHMM training on small problems
"""

import torch
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.chmm_torch import CHMM_torch
from env_adapters.room_adapter import RoomAdapter
from agent_adapters.agent_2d import Agent2D

def test_device_performance(room, seq_len=1000, n_clones_per_obs=30):
    """Test training performance on different devices"""
    
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    devices.append(torch.device("cpu"))
    
    results = {}
    
    for device in devices:
        print(f"\nTesting on {device}:")
        print("-" * 30)
        
        try:
            # Generate trajectory
            print("  Generating trajectory...")
            start_time = time.time()
            env = RoomAdapter(room, seed=42)
            agent = Agent2D(seed=42)
            observations, actions, path = agent.traverse(env, seq_len)
            traj_time = time.time() - start_time
            print(f"    Trajectory generation: {traj_time:.3f}s")
            
            # Move to target device
            x_torch = observations.to(device)
            a_torch = actions.to(device)
            
            # Setup model
            n_emissions = room.max() + 1
            n_clones = torch.ones(n_emissions, dtype=torch.int64, device=device) * n_clones_per_obs
            
            print("  Initializing model...")
            start_time = time.time()
            model = CHMM_torch(
                n_clones=n_clones,
                x=x_torch,
                a=a_torch,
                pseudocount=0.01,
                seed=42,
                device=device
            )
            init_time = time.time() - start_time
            print(f"    Model initialization: {init_time:.3f}s")
            
            # Test single EM iteration
            print("  Running single EM iteration...")
            start_time = time.time()
            
            # Prepare workspace
            if not hasattr(model, '_workspace'):
                model._workspace = {}
            model._T_transposed = model.T.permute(0, 2, 1).contiguous()
            
            # Forward pass
            fwd_start = time.time()
            from models.train_utils import forward
            log2_lik, mess_fwd = forward(
                model._T_transposed, 
                model.Pi_x, 
                model.n_clones,
                x_torch, 
                a_torch, 
                model.device, 
                store_messages=True,
                workspace=model._workspace
            )
            fwd_time = time.time() - fwd_start
            
            # Backward pass
            bwd_start = time.time()
            from models.train_utils import backward
            mess_bwd = backward(
                model.T, 
                model.n_clones, 
                x_torch, 
                a_torch, 
                model.device,
                workspace=model._workspace
            )
            bwd_time = time.time() - bwd_start
            
            # UpdateC pass
            upd_start = time.time()
            from models.train_utils import updateC
            updateC(
                model.C, 
                model.T, 
                model.n_clones, 
                mess_fwd, 
                mess_bwd, 
                x_torch, 
                a_torch, 
                model.device,
                workspace=model._workspace
            )
            upd_time = time.time() - upd_start
            
            # Update T
            t_start = time.time()
            model.update_T()
            t_time = time.time() - t_start
            
            total_time = time.time() - start_time
            
            print(f"    Forward:  {fwd_time:.3f}s")
            print(f"    Backward: {bwd_time:.3f}s") 
            print(f"    UpdateC:  {upd_time:.3f}s")
            print(f"    Update T: {t_time:.3f}s")
            print(f"    Total:    {total_time:.3f}s")
            
            steps_per_sec = seq_len / total_time
            print(f"    Speed: {steps_per_sec:.0f} steps/sec")
            
            # Estimate 100 iterations
            estimated_100_iter = total_time * 100
            print(f"    Est. 100 iterations: {estimated_100_iter:.1f}s")
            
            results[str(device)] = {
                'forward_time': fwd_time,
                'backward_time': bwd_time,
                'updatec_time': upd_time,
                'total_time': total_time,
                'steps_per_sec': steps_per_sec,
                'est_100_iter': estimated_100_iter
            }
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results[str(device)] = {'error': str(e)}
    
    return results

def main():
    print("CPU vs GPU PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Load 5x5 room
    room_path = Path(__file__).parent / "room" / "room_5x5.npy"
    if not room_path.exists():
        print("Creating synthetic 5x5 room...")
        room = np.random.randint(0, 16, (5, 5))
    else:
        room = np.load(room_path)
    
    print(f"Room shape: {room.shape}")
    print(f"Observations: {room.min()} to {room.max()}")
    
    # Test different sequence lengths
    seq_lengths = [1000, 5000, 15000]
    
    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"SEQUENCE LENGTH: {seq_len}")
        print(f"{'='*60}")
        
        results = test_device_performance(room, seq_len, n_clones_per_obs=30)
        
        # Summary
        print(f"\nSUMMARY for {seq_len} steps:")
        print("-" * 40)
        
        best_device = None
        best_time = float('inf')
        
        for device, result in results.items():
            if 'error' not in result:
                est_time = result['est_100_iter']
                print(f"{device:>8}: {est_time:>6.1f}s for 100 iterations")
                if est_time < best_time:
                    best_time = est_time
                    best_device = device
        
        if best_device:
            print(f"WINNER: {best_device} ({best_time:.1f}s)")
            
            # Compare to CPU numba target (10s for 100 iterations)
            if best_time > 10:
                slowdown = best_time / 10
                print(f"❌ {slowdown:.1f}x SLOWER than CPU numba target")
            else:
                speedup = 10 / best_time
                print(f"✅ {speedup:.1f}x FASTER than CPU numba target")

if __name__ == "__main__":
    main()