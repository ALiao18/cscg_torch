"""
Debug test for colab imports
"""

import torch
import numpy as np

def debug_test():
    print("=== Debugging Colab Imports ===")
    
    print(f"1. Basic imports successful")
    
    # Test import
    try:
        from colab_imports_fixed import (
            CSCGEnvironmentAdapter, RoomTorchAdapter, 
            create_room_adapter, get_room_n_clones, CHMM_torch
        )
        print("2. ✓ colab_imports_fixed imported")
        
        print("3. ✓ Functions available")
        
        # Test base adapter directly
        try:
            base = CSCGEnvironmentAdapter()
            print(f"4. ✓ Base adapter created, device: {base.device}")
        except Exception as e:
            print(f"4. ✗ Base adapter failed: {e}")
        
        # Test room adapter creation
        room_tensor = torch.randint(0, 4, size=[5, 5], dtype=torch.long)
        print(f"5. ✓ Room tensor created: {room_tensor.shape}")
        
        # Test adapter creation step by step
        print("6. Testing RoomTorchAdapter creation...")
        try:
            env = RoomTorchAdapter(room_tensor)
            print(f"   ✓ RoomTorchAdapter created successfully!")
            print(f"   ✓ Device: {env.device}")
            print(f"   ✓ Room shape: {env.room.shape}")
        except Exception as e:
            print(f"   ✗ RoomTorchAdapter failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test create_room_adapter function
        print("7. Testing create_room_adapter function...")
        try:
            env2 = create_room_adapter(room_tensor)
            print(f"   ✓ create_room_adapter successful!")
            print(f"   ✓ Device: {env2.device}")
        except Exception as e:
            print(f"   ✗ create_room_adapter failed: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_test()