"""
Test script for Colab imports
"""

def test_colab_imports():
    """Test that colab imports work correctly."""
    import torch
    
    # Test the imports
    from colab_imports import create_room_adapter, get_room_n_clones, CHMM_torch
    
    print("✓ Imports successful")
    
    # Test room creation
    room_tensor = torch.randint(0, 4, size=[5, 5], dtype=torch.long)
    print(f"✓ Room tensor created: {room_tensor.shape}")
    
    # Test adapter creation
    env = create_room_adapter(room_tensor)
    print(f"✓ Environment created: {type(env).__name__}")
    print(f"✓ Environment device: {env.device}")
    
    # Test n_clones
    n_clones = get_room_n_clones(1)
    print(f"✓ n_clones created: {n_clones.shape}")
    
    # Test sequence generation
    x, a = env.generate_sequence(10)
    print(f"✓ Sequences generated: x={len(x)}, a={len(a)}")
    
    # Test model creation
    model = CHMM_torch(n_clones, torch.tensor(x), torch.tensor(a))
    print(f"✓ Model created on: {model.device}")
    
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    test_colab_imports()