#!/usr/bin/env python3
"""
Concise CSCG training script for 20x20 room
"""
import torch
import numpy as np
from tqdm import trange

from cscg_torch.models.train_utils import train_chmm
from cscg_torch.env_adapters.room_utils import create_room_adapter, get_room_n_clones

def main():
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load your room data (assuming it's saved as room_20x20.npy)
    room_data = np.array([
        [ 6, 10,  7,  6, 15, 12, 13, 10,  2,  8,  3,  2,  2,  8,  0, 10, 14, 14, 11, 11],
        [ 7, 15, 10,  3,  8,  1,  7,  6,  3,  9, 15, 12,  3,  9, 14,  8, 10,  1,  1,  4],
        [13,  8,  6, 13,  3,  9, 12,  4,  9, 12,  3,  3, 13,  4,  3, 11, 10, 11,  7,  3],
        [ 9,  1,  6,  7, 14, 15,  3,  8,  0,  8,  6, 13, 12,  5,  4,  1,  6, 13,  5,  5],
        [13, 13,  8,  1,  8,  8,  5,  4, 15,  8,  7,  7,  7,  0,  1,  4,  5, 11, 10,  1],
        [11,  1, 15,  0, 15,  8,  1,  4, 14,  5, 13,  9, 15,  9,  1,  7,  0,  9,  8,  6],
        [11,  6,  7, 13,  8,  5, 15,  0, 12,  0,  2, 12,  6,  7, 15,  8,  7,  6, 15, 15],
        [ 6, 14,  9,  5, 12,  8,  6,  4, 14, 14, 13, 13,  3, 12,  0, 10,  5,  4,  0,  6],
        [ 9, 10,  6,  7,  4, 11,  0,  7,  7,  8, 14, 13,  2,  6,  2, 12,  6,  1, 15, 13],
        [12,  7, 15,  6,  3, 10, 11,  6,  4,  7,  9,  2,  0,  9,  1,  8,  4,  3,  0, 15],
        [ 6, 14,  0, 10, 13,  3, 13,  4,  4, 10, 11,  9, 12,  1, 12,  6, 12,  6,  4, 11],
        [ 2, 15,  6,  3, 13, 12,  7,  6,  0,  8,  3,  9, 13,  0,  0,  8,  2, 12, 13, 12],
        [15,  0,  1,  4,  6,  6, 14, 11,  1, 14,  6,  4,  4,  4, 13, 10,  5,  2,  0,  5],
        [12, 13,  4,  3,  5,  2, 13, 12,  8, 14, 10,  6,  9,  6, 14, 12, 14,  9, 11, 10],
        [ 9,  8,  6,  7,  4,  0,  1,  1,  1,  3,  3, 12, 10,  4,  6, 12,  7,  4,  7, 12],
        [ 2, 13, 12, 12, 11,  6,  2, 13,  8,  4,  1, 12,  4, 11,  8, 12,  1,  8,  3,  3],
        [ 1,  4,  9, 15,  2,  6,  2, 13,  0,  7,  5,  1, 12, 10,  4,  1,  6,  4,  4, 14],
        [ 5, 13, 13,  0, 13,  2,  6,  2,  2, 14,  7,  2, 11,  8,  0,  8,  3,  9,  5, 15],
        [ 4,  6, 12,  9,  8, 10, 15,  1, 15, 10,  4,  2,  7, 15,  4,  9,  6, 10, 15, 13],
        [ 4, 12, 14,  2, 11,  7,  5, 10,  0,  2,  1, 13,  5, 14,  8,  6,  6,  0, 15, 13]
    ])
    
    # Add walls around the room (adapter expects -1 for walls)
    room_with_walls = np.full((22, 22), -1, dtype=np.int64)  # Add 2x2 border
    room_with_walls[1:-1, 1:-1] = room_data  # Place room data in center
    
    # Convert to tensor on correct device
    room_tensor = torch.tensor(room_with_walls, device=device, dtype=torch.int64)
    adapter = create_room_adapter(room_tensor, adapter_type="torch", seed=42)
    print(f"Room shape: {room_data.shape}")
    
    # Generate training sequence
    seq_len = 5000
    print(f"Generating sequence of length {seq_len}...")
    x_seq, a_seq = adapter.generate_sequence(seq_len)
    
    # Convert to tensors
    x = torch.tensor(x_seq, device=device, dtype=torch.int64)
    a = torch.tensor(a_seq, device=device, dtype=torch.int64)
    
    # Setup model parameters
    n_clones = get_room_n_clones(n_clones_per_obs=2, device=device)
    print(f"Training with {len(n_clones)} total clones")
    
    # Train model
    print("Training CSCG model...")
    model, progression = train_chmm(
        n_clones, x, a,
        device=device,
        method='em_T',
        n_iter=100,
        pseudocount=0.01,
        early_stopping=True,
        patience=10
    )
    
    # Evaluate
    final_bps = model.bps(x, a, reduce=True)
    print(f"Final bits per symbol: {final_bps:.4f}")
    
    # Decode states
    print("Decoding optimal state sequence...")
    _, states = model.decode(x, a)
    
    print("Training completed!")
    return model, progression, final_bps

if __name__ == "__main__":
    model, progression, bps = main()