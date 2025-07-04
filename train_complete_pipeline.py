#!/usr/bin/env python3
"""
Complete CSCG-Torch Training Pipeline

This script ties everything together:
1. Load room from room/ directory
2. Generate action/observation sequences using the room
3. Initialize CHMM with proper clone configuration  
4. Train using EM and Viterbi algorithms
5. Save results and performance metrics

Configuration is loaded from config.py - edit that file to change parameters.
"""

import numpy as np
import torch
import time
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.chmm_torch import CHMM_torch
from env_adapters.room_adapter import RoomAdapter
from agent_adapters.agent_2d import Agent2D
import config


def load_room(room_name):
    """Load room from room/ directory"""
    room_path = Path(__file__).parent / "room" / f"{room_name}.npy"
    if not room_path.exists():
        available_rooms = list((Path(__file__).parent / "room").glob("*.npy"))
        available_names = [r.stem for r in available_rooms]
        raise FileNotFoundError(f"Room '{room_name}' not found. Available: {available_names}")
    
    room = np.load(room_path)
    print(f"Loaded room: {room_name}")
    print(f"  Shape: {room.shape}")
    print(f"  Observation range: [{room.min()}, {room.max()}]")
    return room


def generate_trajectory(room, length=10000, seed=42, debug=False):
    """Generate action/observation trajectory using the adapter architecture"""
    print(f"Generating trajectory with length {length}...")
    
    # Enable debug mode if requested
    if debug:
        from env_adapters.room_adapter import set_debug_mode as set_env_debug
        from agent_adapters.agent_2d import set_debug_mode as set_agent_debug
        set_env_debug(True)
        set_agent_debug(True)
        print("  Debug mode enabled")
    
    # Create environment adapter
    env = RoomAdapter(room, seed=seed)
    
    # Create agent
    agent = Agent2D(seed=seed)
    
    # Generate trajectory using the proper architecture
    observations, actions, path = agent.traverse(env, length)
    
    # Convert to numpy for compatibility
    actions_np = actions.cpu().numpy()
    observations_np = observations.cpu().numpy()
    path_np = np.array(path)
    
    print(f"  Actions shape: {actions_np.shape}, range: [{actions_np.min()}, {actions_np.max()}]")
    print(f"  Observations shape: {observations_np.shape}, range: [{observations_np.min()}, {observations_np.max()}]")
    print(f"  Path shape: {path_np.shape}")
    print(f"  Generated on device: {env.device}")
    
    # Validate the trajectory format for CHMM
    assert len(actions_np) == len(observations_np), f"Actions and observations length mismatch: {len(actions_np)} vs {len(observations_np)}"
    assert len(path_np) == len(observations_np), f"Path and observations length mismatch: {len(path_np)} vs {len(observations_np)}"
    
    return actions_np, observations_np, path_np


def main():
    # Validate and print configuration
    config.validate_config()
    if config.VERBOSE:
        config.print_config()
    
    print("="*80)
    print("CSCG-TORCH COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    # Device selection
    if config.DEVICE == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.DEVICE)
    
    print(f"Using device: {device}")
    
    # Set environment variables for performance optimization
    if config.SKIP_DEVICE_CHECKS:
        os.environ['CHMM_SKIP_DEVICE_CHECKS'] = '1'
    if config.ENABLE_BATCHED_UPDATES:
        os.environ['CHMM_BATCHED_UPDATES'] = '1'
    
    # Step 1: Load room
    print("\n1. LOADING ROOM")
    print("-" * 40)
    room = load_room(config.ROOM_NAME)
    
    # Step 2: Calculate n_emissions
    print("\n2. CALCULATING EMISSIONS")
    print("-" * 40)
    n_emissions = room.max() + 1
    print(f"n_emissions = room.max() + 1 = {room.max()} + 1 = {n_emissions}")
    
    # Step 3: Generate trajectory
    print("\n3. GENERATING TRAJECTORY")
    print("-" * 40)
    traj_time_start = time.time()
    actions, observations, path = generate_trajectory(room, length=config.TRAJECTORY_LENGTH, seed=config.SEED, debug=config.DEBUG_MODE)
    traj_time_end = time.time()
    print(f"Trajectory generation completed in {traj_time_end - traj_time_start:.2f}s")
    
    # Convert to torch tensors
    x_torch = torch.from_numpy(observations).to(device)
    a_torch = torch.from_numpy(actions).to(device)
    
    # Step 4: Setup n_clones
    print("\n4. SETTING UP CLONES")
    print("-" * 40)
    n_clones = torch.ones(n_emissions, dtype=torch.int64, device=device) * config.N_CLONES_PER_OBS
    total_states = n_clones.sum().item()
    print(f"n_clones: {config.N_CLONES_PER_OBS} clones per observation")
    print(f"Total states: {n_emissions} emissions × {config.N_CLONES_PER_OBS} clones = {total_states}")
    
    # Step 5: Initialize model
    print("\n5. INITIALIZING MODEL")
    print("-" * 40)
    print(f"Pseudocount: {config.PSEUDOCOUNT}")
    print(f"Seed: {config.SEED}")
    
    model = CHMM_torch(
        n_clones=n_clones,
        x=x_torch,
        a=a_torch,
        pseudocount=config.PSEUDOCOUNT,
        seed=config.SEED,
        device=device
    )
    
    print(f"Model initialized:")
    print(f"  Transition matrix T: {model.T.shape}")
    print(f"  Count matrix C: {model.C.shape}")
    print(f"  Initial state distribution Pi_x: {model.Pi_x.shape}")
    
    # Step 6: Train with EM
    print("\n6. EM TRAINING")
    print("-" * 40)
    
    start_time = time.time()
    em_convergence = model.learn_em_T(x_torch, a_torch, n_iter=config.EM_ITERATIONS, term_early=config.TERM_EARLY)
    em_time = time.time() - start_time
    
    print(f"EM training completed in {em_time:.2f}s")
    print(f"Final EM BPS: {em_convergence[-1]:.6f}")
    print(f"EM convergence length: {len(em_convergence)} iterations")
    
    # Step 7: Train with Viterbi
    print("\n7. VITERBI TRAINING")
    print("-" * 40)
    
    start_time = time.time()
    viterbi_convergence = model.learn_viterbi_T(x_torch, a_torch, n_iter=config.VITERBI_ITERATIONS)
    viterbi_time = time.time() - start_time
    
    print(f"Viterbi training completed in {viterbi_time:.2f}s")
    print(f"Final Viterbi BPS: {viterbi_convergence[-1]:.6f}")
    print(f"Viterbi convergence length: {len(viterbi_convergence)} iterations")
    
    # Step 8: Performance summary
    print("\n8. PERFORMANCE SUMMARY")
    print("-" * 40)
    
    total_time = em_time + viterbi_time
    em_steps_per_sec = config.TRAJECTORY_LENGTH * len(em_convergence) / em_time
    viterbi_steps_per_sec = config.TRAJECTORY_LENGTH * len(viterbi_convergence) / viterbi_time
    
    print(f"Room: {config.ROOM_NAME} ({room.shape})")
    print(f"Trajectory length: {config.TRAJECTORY_LENGTH}")
    print(f"Total states: {total_states}")
    print(f"Device: {device}")
    print()
    print(f"EM Training:")
    print(f"  Time: {em_time:.2f}s")
    print(f"  Iterations: {len(em_convergence)}")
    print(f"  Speed: {em_steps_per_sec:.0f} steps/sec")
    print(f"  Final BPS: {em_convergence[-1]:.6f}")
    print()
    print(f"Viterbi Training:")
    print(f"  Time: {viterbi_time:.2f}s") 
    print(f"  Iterations: {len(viterbi_convergence)}")
    print(f"  Speed: {viterbi_steps_per_sec:.0f} steps/sec")
    print(f"  Final BPS: {viterbi_convergence[-1]:.6f}")
    print()
    print(f"Total time: {total_time:.2f}s")
    
    # Step 9: Save results (optional)
    if config.SAVE_RESULTS:
        print("\n9. SAVING RESULTS")
        print("-" * 40)
        
        results_dir = Path(__file__).parent / config.RESULTS_BASE_DIR / config.get_experiment_name()
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(model.state_dict(), results_dir / "em_model.pt")
        torch.save(model_viterbi.state_dict(), results_dir / "viterbi_model.pt")
        
        # Save convergence data
        np.save(results_dir / "em_convergence.npy", np.array(em_convergence))
        np.save(results_dir / "viterbi_convergence.npy", np.array(viterbi_convergence))
        
        # Save trajectory data
        np.save(results_dir / "actions.npy", actions)
        np.save(results_dir / "observations.npy", observations)
        np.save(results_dir / "path.npy", path)
        
        # Save experiment config
        experiment_config = {
            'room': config.ROOM_NAME,
            'room_shape': room.shape,
            'length': config.TRAJECTORY_LENGTH,
            'n_emissions': n_emissions,
            'n_clones_per_obs': config.N_CLONES_PER_OBS,
            'total_states': total_states,
            'pseudocount': config.PSEUDOCOUNT,
            'seed': config.SEED,
            'device': str(device),
            'em_iter': config.EM_ITERATIONS,
            'viterbi_iter': config.VITERBI_ITERATIONS,
            'em_time': em_time,
            'viterbi_time': viterbi_time,
            'em_final_bps': em_convergence[-1],
            'viterbi_final_bps': viterbi_convergence[-1],
            'debug_mode': config.DEBUG_MODE,
            'batched_updates': config.ENABLE_BATCHED_UPDATES,
        }
        
        import json
        with open(results_dir / "config.json", 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        print(f"Results saved to: {results_dir}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()