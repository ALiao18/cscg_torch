#!/usr/bin/env python3
"""
Complete CSCG_Torch Full Run Example

This script demonstrates the complete workflow:
1. Create room environment
2. Generate large sequences (150k steps)
3. Train CHMM model
4. Evaluate and analyze results

This is a production-ready example showing exactly how to use the system.
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

def create_complex_room(size=50, complexity=0.3, seed=42):
    """
    Create a complex room with walls, corridors, and open areas.
    
    Args:
        size (int): Room dimensions (size x size)
        complexity (float): 0-1, how complex the room layout is
        seed (int): Random seed for reproducibility
    
    Returns:
        torch.Tensor: Room layout tensor
    """
    print(f"🏗️  Creating {size}x{size} room (complexity: {complexity})")
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Start with open space
    room = torch.zeros(size, size, dtype=torch.long, device=device)
    
    # Add boundary walls
    room[0, :] = -1  # Top wall
    room[-1, :] = -1  # Bottom wall
    room[:, 0] = -1  # Left wall
    room[:, -1] = -1  # Right wall
    
    # Add internal structure based on complexity
    if complexity > 0:
        # Add some internal walls
        num_walls = int(size * complexity)
        for _ in range(num_walls):
            # Random wall placement
            if torch.rand(1) < 0.5:  # Horizontal wall
                row = torch.randint(2, size-2, (1,)).item()
                start_col = torch.randint(1, size//2, (1,)).item()
                end_col = start_col + torch.randint(3, size//3, (1,)).item()
                end_col = min(end_col, size-1)
                room[row, start_col:end_col] = -1
            else:  # Vertical wall
                col = torch.randint(2, size-2, (1,)).item()
                start_row = torch.randint(1, size//2, (1,)).item()
                end_row = start_row + torch.randint(3, size//3, (1,)).item()
                end_row = min(end_row, size-1)
                room[start_row:end_row, col] = -1
    
    # Add some different floor types for variety
    for floor_type in range(1, 4):
        mask = torch.rand(size, size, device=device) < 0.1
        room = torch.where((room == 0) & mask, floor_type, room)
    
    # Ensure there are always some free spaces
    free_spaces = (room != -1).sum()
    if free_spaces < size * size * 0.3:  # At least 30% free space
        # Remove some walls to ensure navigability
        wall_positions = (room == -1) & (torch.rand(size, size, device=device) < 0.3)
        room = torch.where(wall_positions, 0, room)
    
    print(f"✅ Room created: {((room != -1).sum().item())} free spaces out of {size*size}")
    return room

def generate_training_data(room_tensor, sequence_length=150000, adapter_type="torch", seed=42):
    """
    Generate training sequences from the room environment.
    
    Args:
        room_tensor (torch.Tensor): Room layout
        sequence_length (int): Number of steps to generate
        adapter_type (str): "torch" or "numpy"
        seed (int): Random seed
    
    Returns:
        tuple: (observations, actions) as numpy arrays
    """
    print(f"\n🎮 Generating {sequence_length} training steps...")
    
    from env_adapters.room_utils import create_room_adapter
    
    # Create environment adapter
    adapter = create_room_adapter(room_tensor, adapter_type=adapter_type)
    adapter.rng = np.random.RandomState(seed)  # Set seed for reproducibility
    
    # Generate sequences
    start_time = time.time()
    x_seq, a_seq = adapter.generate_sequence(sequence_length)
    elapsed = time.time() - start_time
    
    print(f"✅ Generated {len(x_seq)} steps in {elapsed:.2f}s ({len(x_seq)/elapsed:.0f} steps/sec)")
    print(f"   Observation range: [{x_seq.min()}, {x_seq.max()}]")
    print(f"   Action range: [{a_seq.min()}, {a_seq.max()}]")
    print(f"   Data types: x={x_seq.dtype}, a={a_seq.dtype}")
    print(f"   Return types: x={type(x_seq)}, a={type(a_seq)}")
    
    # Validate the generated data
    assert isinstance(x_seq, np.ndarray), f"x_seq must be numpy array, got {type(x_seq)}"
    assert isinstance(a_seq, np.ndarray), f"a_seq must be numpy array, got {type(a_seq)}"
    assert len(x_seq) == len(a_seq), "Sequence lengths must match"
    assert len(x_seq) > 0, "Sequences cannot be empty"
    
    return x_seq, a_seq

def create_and_train_chmm(x_seq, a_seq, n_clones_per_obs=3, n_em_iterations=20, seed=42):
    """
    Create and train a CHMM model on the generated sequences.
    
    Args:
        x_seq (np.ndarray): Observation sequence
        a_seq (np.ndarray): Action sequence  
        n_clones_per_obs (int): Number of clones per observation type
        n_em_iterations (int): Number of EM training iterations
        seed (int): Random seed
    
    Returns:
        tuple: (model, convergence_history)
    """
    print(f"\n🧠 Creating and training CHMM model...")
    
    from models.chmm_torch import CHMM_torch
    from env_adapters.room_utils import get_room_n_clones
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # Convert sequences to tensors
    print("   Converting sequences to tensors...")
    x_tensor = torch.tensor(x_seq, dtype=torch.int64, device=device)
    a_tensor = torch.tensor(a_seq, dtype=torch.int64, device=device)
    
    # Create n_clones for room navigation (16 possible observations: 2^4)
    n_clones = get_room_n_clones(n_clones_per_obs=n_clones_per_obs, device=device)
    print(f"   n_clones shape: {n_clones.shape}, total states: {n_clones.sum()}")
    
    # Create CHMM model
    print("   Creating CHMM model...")
    start_time = time.time()
    model = CHMM_torch(n_clones, x_tensor, a_tensor, seed=seed)
    creation_time = time.time() - start_time
    
    print(f"✅ Model created in {creation_time:.2f}s")
    print(f"   Model device: {model.device}")
    print(f"   Transition matrix shape: {model.T.shape}")
    print(f"   Number of actions: {model.T.shape[0]}")
    print(f"   Number of states: {model.T.shape[1]}")
    
    # Compute initial BPS
    print("   Computing initial bits-per-step...")
    initial_bps = model.bps(x_tensor, a_tensor)
    print(f"   Initial BPS: {initial_bps.item():.4f}")
    
    # Train with EM algorithm
    print(f"   Training with EM algorithm ({n_em_iterations} iterations)...")
    start_time = time.time()
    convergence = model.learn_em_T(x_tensor, a_tensor, n_iter=n_em_iterations, term_early=True)
    training_time = time.time() - start_time
    
    final_bps = convergence[-1]
    improvement = initial_bps.item() - final_bps
    
    print(f"✅ Training completed in {training_time:.2f}s")
    print(f"   Final BPS: {final_bps:.4f}")
    print(f"   Improvement: {improvement:.4f} bits/step")
    print(f"   Convergence steps: {len(convergence)}")
    
    return model, convergence

def analyze_results(model, x_seq, a_seq, convergence, room_tensor):
    """
    Analyze and visualize the training results.
    
    Args:
        model: Trained CHMM model
        x_seq (np.ndarray): Original observation sequence
        a_seq (np.ndarray): Original action sequence
        convergence (list): Training convergence history
        room_tensor (torch.Tensor): Original room layout
    """
    print(f"\n📊 Analyzing results...")
    
    device = model.device
    x_tensor = torch.tensor(x_seq, dtype=torch.int64, device=device)
    a_tensor = torch.tensor(a_seq, dtype=torch.int64, device=device)
    
    # Compute final statistics
    final_bps = model.bps(x_tensor, a_tensor)
    print(f"   Final BPS: {final_bps.item():.4f}")
    
    # Analyze observation distribution
    obs_counts = np.bincount(x_seq, minlength=16)
    print(f"   Observation distribution:")
    for i, count in enumerate(obs_counts):
        if count > 0:
            print(f"     Obs {i:2d}: {count:6d} steps ({count/len(x_seq)*100:.1f}%)")
    
    # Analyze action distribution
    action_counts = np.bincount(a_seq, minlength=4)
    action_names = ["Up", "Down", "Left", "Right"]
    print(f"   Action distribution:")
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        print(f"     {name:5s}: {count:6d} steps ({count/len(a_seq)*100:.1f}%)")
    
    # Analyze room statistics
    room_stats = {
        "Total cells": room_tensor.numel(),
        "Free spaces": (room_tensor != -1).sum().item(),
        "Walls": (room_tensor == -1).sum().item(),
        "Room size": f"{room_tensor.shape[0]}x{room_tensor.shape[1]}"
    }
    print(f"   Room statistics:")
    for key, value in room_stats.items():
        print(f"     {key}: {value}")
    
    # Generate some predictions
    print(f"\n🎯 Testing model predictions...")
    
    # Test on a subset for speed
    test_size = min(10000, len(x_seq))
    x_test = x_tensor[:test_size]
    a_test = a_tensor[:test_size]
    
    test_bps = model.bps(x_test, a_test)
    print(f"   Test BPS (first {test_size} steps): {test_bps.item():.4f}")
    
    # Decode most likely state sequence
    print("   Computing MAP state sequence...")
    map_likelihood, map_states = model.decode(x_test, a_test)
    print(f"   MAP likelihood: {map_likelihood.item():.4f}")
    print(f"   MAP states shape: {map_states.shape}")
    print(f"   State distribution: min={map_states.min()}, max={map_states.max()}")
    
    return {
        'final_bps': final_bps.item(),
        'convergence': convergence,
        'obs_counts': obs_counts,
        'action_counts': action_counts,
        'room_stats': room_stats,
        'test_bps': test_bps.item(),
        'map_likelihood': map_likelihood.item()
    }

def save_results(results, convergence, room_tensor, save_dir="results"):
    """
    Save results and create visualizations.
    
    Args:
        results (dict): Analysis results
        convergence (list): Training convergence history
        room_tensor (torch.Tensor): Room layout
        save_dir (str): Directory to save results
    """
    print(f"\n💾 Saving results to {save_dir}/...")
    
    # Create results directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # Save convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(convergence, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('CHMM Training Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('EM Iteration')
    plt.ylabel('Bits per Step (BPS)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    convergence_path = Path(save_dir) / "convergence.png"
    plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Convergence plot saved: {convergence_path}")
    
    # Save room visualization
    plt.figure(figsize=(8, 8))
    room_np = room_tensor.cpu().numpy()
    plt.imshow(room_np, cmap='viridis', interpolation='nearest')
    plt.title('Room Layout', fontsize=14, fontweight='bold')
    plt.colorbar(label='Cell Type (-1=Wall, 0-3=Floor)')
    plt.tight_layout()
    room_path = Path(save_dir) / "room_layout.png"
    plt.savefig(room_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Room layout saved: {room_path}")
    
    # Save observation distribution
    plt.figure(figsize=(10, 6))
    obs_indices = np.arange(16)
    plt.bar(obs_indices, results['obs_counts'], alpha=0.7, color='skyblue', edgecolor='navy')
    plt.title('Observation Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Observation Type')
    plt.ylabel('Count')
    plt.xticks(obs_indices)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    obs_path = Path(save_dir) / "observation_distribution.png"
    plt.savefig(obs_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Observation distribution saved: {obs_path}")
    
    # Save text summary
    summary_path = Path(save_dir) / "results_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CSCG_Torch Training Results Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Final BPS: {results['final_bps']:.4f}\n")
        f.write(f"Training iterations: {len(convergence)}\n")
        f.write(f"Improvement: {convergence[0] - results['final_bps']:.4f} bits/step\n\n")
        f.write("Room Statistics:\n")
        for key, value in results['room_stats'].items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nTest BPS: {results['test_bps']:.4f}\n")
        f.write(f"MAP likelihood: {results['map_likelihood']:.4f}\n")
    
    print(f"   ✅ Summary saved: {summary_path}")

def main():
    """
    Complete full run example of the CSCG_Torch system.
    """
    print("🚀 CSCG_TORCH FULL RUN EXAMPLE")
    print("="*60)
    print("This example demonstrates the complete workflow:")
    print("1. Create complex room environment")
    print("2. Generate large training sequences (150k steps)")
    print("3. Train CHMM model with EM algorithm")
    print("4. Analyze and save results")
    print("="*60)
    
    # Configuration
    config = {
        'room_size': 50,
        'room_complexity': 0.2,
        'sequence_length': 150000,
        'n_clones_per_obs': 3,
        'em_iterations': 25,
        'seed': 42
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Step 1: Create room environment
        room_tensor = create_complex_room(
            size=config['room_size'],
            complexity=config['room_complexity'],
            seed=config['seed']
        )
        
        # Step 2: Generate training data
        x_seq, a_seq = generate_training_data(
            room_tensor=room_tensor,
            sequence_length=config['sequence_length'],
            adapter_type="torch",
            seed=config['seed']
        )
        
        # Step 3: Train CHMM model
        model, convergence = create_and_train_chmm(
            x_seq=x_seq,
            a_seq=a_seq,
            n_clones_per_obs=config['n_clones_per_obs'],
            n_em_iterations=config['em_iterations'],
            seed=config['seed']
        )
        
        # Step 4: Analyze results
        results = analyze_results(model, x_seq, a_seq, convergence, room_tensor)
        
        # Step 5: Save results
        save_results(results, convergence, room_tensor)
        
        # Final summary
        print(f"\n🎉 FULL RUN COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ Room: {config['room_size']}x{config['room_size']} with {results['room_stats']['Free spaces']} free spaces")
        print(f"✅ Data: {len(x_seq)} training steps generated")
        print(f"✅ Model: {model.T.shape[1]} states, {model.T.shape[0]} actions")
        print(f"✅ Training: {len(convergence)} EM iterations")
        print(f"✅ Performance: {results['final_bps']:.4f} final BPS")
        print(f"✅ Improvement: {convergence[0] - results['final_bps']:.4f} bits/step")
        print(f"✅ Results saved to 'results/' directory")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ FULL RUN FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)