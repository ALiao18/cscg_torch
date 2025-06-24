"""
Colab Diagnostic Script

This script helps diagnose the exact issue happening in Google Colab
"""

import torch
import numpy as np

def diagnose_colab_issue():
    print("=== COLAB DIAGNOSTIC ===")
    
    # Test 1: Basic tensor operations
    print("1. Testing basic tensor operations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    room = torch.randint(0, 4, size=[3, 3], dtype=torch.long, device=device)
    print(f"   Room tensor: {room.shape}, device: {room.device}")
    
    # Test 2: Test tensor indexing and conversion
    print("2. Testing tensor indexing...")
    r, c = 1, 1
    val = room[r, c]
    print(f"   room[{r}, {c}] = {val}, type: {type(val)}")
    print(f"   val.item() = {val.item()}, type: {type(val.item())}")
    
    # Test 3: Test tensor comparison
    print("3. Testing tensor comparisons...")
    comparison = room[r, c] != -1
    print(f"   Comparison result: {comparison}, type: {type(comparison)}")
    print(f"   Comparison.item(): {comparison.item()}, type: {type(comparison.item())}")
    
    # Test 4: Test numpy random choice
    print("4. Testing numpy random choice...")
    rng = np.random.RandomState(42)
    n_actions = 4
    action = rng.choice(n_actions)
    print(f"   Action: {action}, type: {type(action)}")
    
    # Test 5: Test with tensor length
    print("5. Testing with tensor operations...")
    free_pos = (room != -1).nonzero(as_tuple=False)
    print(f"   Free positions shape: {free_pos.shape}")
    print(f"   len(free_pos): {len(free_pos)}, type: {type(len(free_pos))}")
    
    if len(free_pos) > 0:
        idx = rng.choice(len(free_pos))
        print(f"   Random idx: {idx}, type: {type(idx)}")
        selected = free_pos[idx]
        print(f"   Selected: {selected}, type: {type(selected)}")
        selected_list = selected.tolist()
        print(f"   Selected list: {selected_list}, type: {type(selected_list)}")
    
    # Test 6: Test the problematic conversion
    print("6. Testing problematic conversions...")
    try:
        int_action = int(action)
        print(f"   int(action) works: {int_action}, type: {type(int_action)}")
    except Exception as e:
        print(f"   int(action) failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    diagnose_colab_issue()