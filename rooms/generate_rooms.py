#!/usr/bin/env python3
"""
Room Generation Script for CSCG Testing

Generates square rooms of different sizes with 16 possible observation states
compatible with the CSCG environment adapters.
"""

import numpy as np
import torch
import random
from pathlib import Path

def generate_room_layout(size: int, num_states: int = 16, seed: int = 42) -> np.ndarray:
    """
    Generate a square room layout with random state assignments.
    
    Args:
        size: Room dimensions (size x size)
        num_states: Number of possible observation states (default: 16)
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Room layout with state assignments
    """
    print(f"Generating {size}x{size} room with {num_states} states...")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize room with random states
    room = np.random.randint(0, num_states, size=(size, size), dtype=np.int64)
    
    # Add some structure to make it more realistic
    # Create walls around the border (state -1 represents walls)
    if size > 2:
        # Border walls
        room[0, :] = -1  # Top wall
        room[-1, :] = -1  # Bottom wall
        room[:, 0] = -1  # Left wall
        room[:, -1] = -1  # Right wall
        
        # Add some internal walls for larger rooms
        if size >= 10:
            # Add some internal structure
            mid = size // 2
            # Add a cross pattern for larger rooms
            room[mid, 1:-1] = -1  # Horizontal line
            room[1:-1, mid] = -1  # Vertical line
            
            # Create openings in the cross
            room[mid, mid-1:mid+2] = np.random.randint(0, num_states, 3)
            room[mid-1:mid+2, mid] = np.random.randint(0, num_states, 3)
        
        elif size >= 5:
            # Add a few random internal walls for medium rooms
            for _ in range(size // 3):
                i, j = np.random.randint(1, size-1, 2)
                if np.random.random() > 0.7:  # 30% chance of wall
                    room[i, j] = -1
    
    print(f"  - Room dimensions: {room.shape}")
    print(f"  - State range: {room.min()} to {room.max()}")
    print(f"  - Free cells: {np.sum(room != -1)}")
    print(f"  - Wall cells: {np.sum(room == -1)}")
    
    return room

def save_room_data(room: np.ndarray, size: int, num_states: int, 
                   output_dir: Path) -> None:
    """
    Save room data in multiple formats for compatibility.
    
    Args:
        room: Room layout array
        size: Room size
        num_states: Number of states
        output_dir: Output directory path
    """
    base_name = f"room_{size}x{size}_{num_states}states"
    
    # Save as NumPy array (for RoomNPAdapter)
    numpy_path = output_dir / f"{base_name}.npy"
    np.save(numpy_path, room)
    print(f"  - Saved NumPy array: {numpy_path}")
    
    # Save as PyTorch tensor (for RoomTorchAdapter)
    tensor_path = output_dir / f"{base_name}.pt"
    room_tensor = torch.from_numpy(room).clone()
    torch.save(room_tensor, tensor_path)
    print(f"  - Saved PyTorch tensor: {tensor_path}")
    
    # Save as text file for human inspection
    txt_path = output_dir / f"{base_name}.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Room Layout: {size}x{size} with {num_states} states\n")
        f.write("=" * 50 + "\n\n")
        f.write("Legend:\n")
        f.write("  -1: Wall\n")
        f.write(f"  0-{num_states-1}: Free cells with different observation states\n\n")
        f.write("Room Layout:\n")
        for i, row in enumerate(room):
            f.write(f"Row {i:2d}: ")
            for cell in row:
                if cell == -1:
                    f.write("## ")
                else:
                    f.write(f"{cell:2d} ")
            f.write("\n")
    print(f"  - Saved text description: {txt_path}")

def create_room_info_file(output_dir: Path, room_sizes: list, num_states: int) -> None:
    """
    Create an information file about the generated rooms.
    """
    info_path = output_dir / "README.md"
    
    with open(info_path, 'w') as f:
        f.write("# CSCG Test Rooms\n\n")
        f.write("This directory contains test room layouts for CSCG environment adapters.\n\n")
        
        f.write("## Room Specifications\n\n")
        f.write(f"- **Number of observation states**: {num_states} (0-{num_states-1})\n")
        f.write("- **Wall representation**: -1\n")
        f.write("- **Data types**: NumPy arrays (.npy) and PyTorch tensors (.pt)\n\n")
        
        f.write("## Available Rooms\n\n")
        for size in room_sizes:
            f.write(f"### {size}x{size} Room\n")
            f.write(f"- **Files**: `room_{size}x{size}_{num_states}states.{{npy,pt,txt}}`\n")
            f.write(f"- **Dimensions**: {size} x {size}\n")
            f.write(f"- **Total cells**: {size * size}\n")
            f.write(f"- **Use case**: {'Fast testing' if size == 5 else 'Standard testing' if size <= 20 else 'Performance testing'}\n\n")
        
        f.write("## Usage Examples\n\n")
        f.write("### Loading with NumPy\n")
        f.write("```python\n")
        f.write("import numpy as np\n")
        f.write("room = np.load('room_5x5_16states.npy')\n")
        f.write("```\n\n")
        
        f.write("### Loading with PyTorch\n")
        f.write("```python\n")
        f.write("import torch\n")
        f.write("room = torch.load('room_5x5_16states.pt')\n")
        f.write("```\n\n")
        
        f.write("### Using with CSCG Adapters\n")
        f.write("```python\n")
        f.write("from cscg_torch.env_adapters.room_utils import create_room_adapter\n")
        f.write("import torch\n\n")
        f.write("# Load room data\n")
        f.write("room = torch.load('room_5x5_16states.pt')\n\n")
        f.write("# Create adapter\n")
        f.write("adapter = create_room_adapter(room, adapter_type='torch')\n\n")
        f.write("# Generate test sequences\n")
        f.write("x_seq, a_seq = adapter.generate_sequence(2000)\n")
        f.write("```\n\n")
        
        f.write("## Compatibility\n\n")
        f.write("- Compatible with `RoomNPAdapter` and `RoomTorchAdapter`\n")
        f.write("- Works with `create_room_adapter()` utility function\n")
        f.write("- Suitable for testing CHMM training and inference\n")

def main():
    """
    Main function to generate all test rooms.
    """
    print("CSCG Room Generation Script")
    print("=" * 50)
    
    # Configuration
    room_sizes = [5, 10, 20, 50]
    num_states = 16
    output_dir = Path(".")
    
    print(f"Generating rooms with {num_states} observation states...")
    print(f"Room sizes: {room_sizes}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    # Generate each room size
    for size in room_sizes:
        print(f"Processing {size}x{size} room...")
        try:
            # Generate room layout
            room = generate_room_layout(size, num_states, seed=42 + size)
            
            # Save in multiple formats
            save_room_data(room, size, num_states, output_dir)
            
            print(f"✓ Successfully generated {size}x{size} room\n")
            
        except Exception as e:
            print(f"✗ Error generating {size}x{size} room: {e}\n")
    
    # Create information file
    try:
        create_room_info_file(output_dir, room_sizes, num_states)
        print("✓ Created README.md with room information")
    except Exception as e:
        print(f"✗ Error creating README.md: {e}")
    
    print("\nRoom generation complete!")
    print(f"Generated {len(room_sizes)} room layouts in {output_dir.absolute()}")

if __name__ == "__main__":
    main()