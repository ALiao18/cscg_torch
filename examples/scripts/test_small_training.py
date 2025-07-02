#!/usr/bin/env python3
"""
Small test to verify training works with proper device handling
"""
import torch
import numpy as np
from models.train_utils import train_chmm
from env_adapters.room_utils import create_room_adapter, get_room_n_clones

def test_small_training():
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create small room data
    room_data = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    
    # Convert to tensor on device
    room_tensor = torch.tensor(room_data, device=device, dtype=torch.int64)
    print(f"Room tensor device: {room_tensor.device}")
    
    # Create adapter
    adapter = create_room_adapter(room_tensor, adapter_type="torch", seed=42)
    print(f"Adapter device: {adapter.device}")
    
    # Generate small sequence
    seq_len = 1000
    print(f"Generating sequence of length {seq_len}...")
    x_seq, a_seq = adapter.generate_sequence(seq_len)
    
    # Convert to tensors on device
    x = torch.tensor(x_seq, device=device, dtype=torch.int64)
    a = torch.tensor(a_seq, device=device, dtype=torch.int64)
    print(f"Sequence tensors: x device={x.device}, a device={a.device}")
    
    # Setup model parameters
    n_clones = get_room_n_clones(n_clones_per_obs=10, device=device)
    print(f"n_clones device: {n_clones.device}")
    print(f"Training with {n_clones.sum().item()} total clones")
    
    # Train small model
    print("Starting training...")
    try:
        model, progression = train_chmm(
            n_clones, x, a,
            device=device,
            method='em_T',
            n_iter=5,  # Small number for testing
            learn_E=False,  # Simpler training
            pseudocount=0.1,
            early_stopping=False
        )
        
        print(f"Training completed successfully!")
        print(f"Model device: {model.device}")
        print(f"Final BPS: {progression[-1]:.4f}")
        print(f"Progression: {progression}")
        
        # Test BPS computation
        bps = model.bps(x, a, reduce=True)
        print(f"BPS computation successful: {bps:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_training()
    print(f"Test {'PASSED' if success else 'FAILED'}")