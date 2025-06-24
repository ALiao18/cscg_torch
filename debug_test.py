"""
Debug test for colab imports
"""

def debug_test():
    print("=== Debugging Colab Imports ===")
    
    import torch
    import numpy as np
    
    print(f"1. Basic imports successful")
    
    # Test import
    try:
        from colab_imports import setup_colab_imports
        print("2. ✓ setup_colab_imports imported")
        
        cscg = setup_colab_imports()
        print("3. ✓ CSCG namespace created")
        
        # Test base adapter directly
        try:
            from colab_imports import CSCGEnvironmentAdapter
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
            env = cscg.RoomTorchAdapter(room_tensor)
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
            env2 = cscg.create_room_adapter(room_tensor)
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