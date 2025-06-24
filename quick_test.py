#!/usr/bin/env python3

def quick_test():
    import torch
    import numpy as np
    
    print("=== Quick Test ===")
    
    # Test basic tensor operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test room tensor
    room = torch.randint(0, 4, size=[3, 3], dtype=torch.long, device=device)
    print(f"Room tensor: {room.shape}, device: {room.device}")
    
    # Test observation extraction
    r, c = 1, 1
    up = int(room[r - 1, c] != -1) if r > 0 else 0
    down = int(room[r + 1, c] != -1) if r < 2 else 0
    left = int(room[r, c - 1] != -1) if c > 0 else 0
    right = int(room[r, c + 1] != -1) if c < 2 else 0
    obs = (up << 3) + (down << 2) + (left << 1) + right
    
    print(f"Observation: {obs}, type: {type(obs)}")
    
    # Test tensor creation from list
    obs_list = [obs, obs+1, obs+2]
    obs_tensor = torch.tensor(obs_list, dtype=torch.int64, device=device)
    print(f"Tensor from list: {obs_tensor}, device: {obs_tensor.device}")
    
    print("✅ Quick test passed!")

if __name__ == "__main__":
    quick_test()