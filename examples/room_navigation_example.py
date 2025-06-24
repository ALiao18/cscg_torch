"""
Room Navigation Example

Demonstrates how to use room adapters with CHMM models for spatial navigation tasks.
"""

import torch
import numpy as np
from cscg_torch import CHMM_torch
from cscg_torch.env_adapters import create_room_adapter, get_room_n_clones

def main():
    """Main example function."""
    print("=== CSCG PyTorch Room Navigation Example ===\n")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Create room tensor (your current approach)
    print("\n1. Creating room layout...")
    obs_values = [0, 1, 2, 3]  # Different room types
    room_tensor = torch.randint(0, len(obs_values), size=[10, 10], dtype=torch.long)
    
    # Add walls around the border
    room_tensor[0, :] = -1    # Top wall
    room_tensor[-1, :] = -1   # Bottom wall  
    room_tensor[:, 0] = -1    # Left wall
    room_tensor[:, -1] = -1   # Right wall
    
    print(f"Room shape: {room_tensor.shape}")
    print(f"Room values: {torch.unique(room_tensor)}")
    
    # Step 2: Create room adapter (FIXED - wrap tensor in adapter)
    print("\n2. Creating room adapter...")
    env = create_room_adapter(room_tensor, adapter_type="torch")
    print(f"✓ Room adapter created successfully")
    print(f"✓ Number of actions: {env.n_actions}")
    
    # Step 3: Get n_clones tensor for model
    print("\n3. Setting up model parameters...")
    n_clones = get_room_n_clones(n_clones_per_obs=2, device=device)
    print(f"✓ n_clones shape: {n_clones.shape}")
    print(f"✓ Total clone states: {n_clones.sum()}")
    
    # Step 4: Generate training data
    print("\n4. Generating training sequences...")
    x_seq, a_seq = env.generate_sequence(500)  # Now this works!
    print(f"✓ Generated {len(x_seq)} observations")
    print(f"✓ Generated {len(a_seq)} actions")
    print(f"✓ Observation range: {x_seq.min()} to {x_seq.max()}")
    print(f"✓ Data types: x={x_seq.dtype}, a={a_seq.dtype}")
    print(f"✓ Devices: x={x_seq.device}, a={a_seq.device}")
    
    # Step 5: Create and train CHMM model
    print("\n5. Training CHMM model...")
    model = CHMM_torch(
        n_clones=n_clones,
        x=x_seq, 
        a=a_seq,
        pseudocount=1e-6,
        dtype=torch.float32
    )
    print(f"✓ Model created on device: {model.device}")
    
    # Train the model
    print("   Starting EM training...")
    convergence = model.learn_em_T(x_seq, a_seq, n_iter=20)
    print(f"✓ Training completed!")
    print(f"✓ Final BPS: {convergence[-1]:.4f}")
    print(f"✓ Improvement: {convergence[0] - convergence[-1]:.4f}")
    
    # Step 6: Test inference
    print("\n6. Testing inference...")
    
    # Test likelihood
    bps = model.bps(x_seq, a_seq)
    print(f"✓ Total BPS: {bps:.4f}")
    
    # Test MAP decoding
    map_lik, map_states = model.decode(x_seq, a_seq)
    print(f"✓ MAP log-likelihood: {-map_lik:.4f}")
    print(f"✓ MAP states range: {map_states.min()} to {map_states.max()}")
    
    # Step 7: Generate new sequences
    print("\n7. Generating new sequences...")
    new_x, new_a = model.sample(100)
    print(f"✓ Generated sequence length: {len(new_x)}")
    print(f"✓ New observations: {new_x[:10]}")
    print(f"✓ New actions: {new_a[:10]}")
    
    print("\n=== Example completed successfully! ===")
    return model, env, convergence

def colab_example():
    """Simplified example for Google Colab."""
    print("=== Google Colab Quick Start ===")
    
    # Your current room generation
    room_tensor = torch.randint(0, 4, size=[50, 50], dtype=torch.long)
    
    # ✅ CORRECT: Wrap in adapter
    env = create_room_adapter(room_tensor)
    n_clones = get_room_n_clones(n_clones_per_obs=1)
    
    # Generate data
    x, a = env.generate_sequence(200)
    
    # Train model
    model = CHMM_torch(n_clones, x, a)
    convergence = model.learn_em_T(x, a, n_iter=10)
    
    print(f"✓ Final BPS: {convergence[-1]:.4f}")
    return model

if __name__ == "__main__":
    # Run main example
    model, env, convergence = main()
    
    # Show usage pattern
    print("\n" + "="*50)
    print("USAGE PATTERN FOR YOUR CODE:")
    print("="*50)
    print("""
# ❌ OLD (doesn't work):
room_tensor = torch.randint(0, len(obs_values), size=[50, 50], dtype=torch.long)
x, a = room_tensor.generate_sequence(100)  # ERROR: tensor has no generate_sequence()

# ✅ NEW (works):
room_tensor = torch.randint(0, len(obs_values), size=[50, 50], dtype=torch.long)
env = create_room_adapter(room_tensor)  # Wrap tensor in adapter
n_clones = get_room_n_clones(n_clones_per_obs=2)
x, a = env.generate_sequence(100)       # Now it works!
model = CHMM_torch(n_clones, x, a)
""")