import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from env_adapters.room_adapter import RoomAdapter, set_debug_mode
from agent_adapters.agent_2d import Agent2D, set_debug_mode as set_agent_debug_mode


# Test room - 20x20 with observations 0-15
room = np.array([
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


def test_room_adapter_basic():
    """Test basic room adapter functionality"""
    print("Testing basic RoomAdapter functionality...")
    
    # Enable debug mode
    set_debug_mode(True)
    set_agent_debug_mode(True)
    
    # Create adapter
    adapter = RoomAdapter(room, seed=42)
    
    # Test basic properties
    assert adapter.room_height == 20, f"Expected height 20, got {adapter.room_height}"
    assert adapter.room_width == 20, f"Expected width 20, got {adapter.room_width}"
    assert adapter.n_actions == 4, f"Expected 4 actions, got {adapter.n_actions}"
    
    # Test reset
    obs = adapter.reset()
    assert isinstance(obs, int), f"Observation should be int, got {type(obs)}"
    assert 0 <= obs <= 15, f"Observation should be 0-15, got {obs}"
    
    # Test position is valid
    row, col = adapter.pos
    assert 0 <= row < 20, f"Row should be 0-19, got {row}"
    assert 0 <= col < 20, f"Col should be 0-19, got {col}"
    
    # Test check_avail_actions
    avail_actions = adapter.check_avail_actions(adapter.pos)
    assert len(avail_actions) > 0, "Should have at least one available action"
    assert all(0 <= a <= 3 for a in avail_actions), f"Actions should be 0-3, got {avail_actions}"
    
    print("✓ Basic functionality tests passed")


def test_agent_traversal():
    """Test agent traversal with 1000 steps"""
    print("Testing Agent2D traversal with 1000 steps...")
    
    # Create adapter and agent
    adapter = RoomAdapter(room, seed=42)
    agent = Agent2D(seed=42)
    
    # Generate 1000 observations and actions
    seq_len = 1000
    observations, actions, path = agent.traverse(adapter, seq_len)
    
    # Validate outputs
    assert observations.shape == (seq_len,), f"Observations shape should be ({seq_len},), got {observations.shape}"
    assert actions.shape == (seq_len,), f"Actions shape should be ({seq_len},), got {actions.shape}"
    assert len(path) == seq_len, f"Path length should be {seq_len}, got {len(path)}"
    
    # Validate data types
    assert observations.dtype == torch.int64, f"Observations should be int64, got {observations.dtype}"
    assert actions.dtype == torch.int64, f"Actions should be int64, got {actions.dtype}"
    
    # Validate observation range
    obs_min, obs_max = observations.min().item(), observations.max().item()
    assert 0 <= obs_min <= 15, f"Min observation out of range: {obs_min}"
    assert 0 <= obs_max <= 15, f"Max observation out of range: {obs_max}"
    
    # Validate action range
    act_min, act_max = actions.min().item(), actions.max().item()
    assert 0 <= act_min <= 3, f"Min action out of range: {act_min}"
    assert 0 <= act_max <= 3, f"Max action out of range: {act_max}"
    
    # Validate path coordinates
    for i, (row, col) in enumerate(path):
        assert 0 <= row < 20, f"Step {i}: Row {row} out of range"
        assert 0 <= col < 20, f"Step {i}: Col {col} out of range"
    
    print(f"✓ Generated {seq_len} steps successfully")
    return observations, actions, path


def test_device_compatibility():
    """Test device compatibility (GPU/MPS/CPU)"""
    print("Testing device compatibility...")
    
    adapter = RoomAdapter(room, seed=42)
    agent = Agent2D(seed=42)
    
    print(f"Using device: {adapter.device}")
    
    # Test short sequence on device
    observations, actions, path = agent.traverse(adapter, 100)
    
    # Verify tensors are on correct device
    assert observations.device == adapter.device, f"Observations device mismatch: {observations.device} vs {adapter.device}"
    assert actions.device == adapter.device, f"Actions device mismatch: {actions.device} vs {adapter.device}"
    
    print(f"✓ Device compatibility test passed on {adapter.device}")


def test_colormap():
    """Test colormap functionality"""
    print("Testing colormap functionality...")
    
    adapter = RoomAdapter(room, seed=42)
    
    # Test automatic inference
    cmap_auto = adapter.get_obs_colormap()
    assert cmap_auto is not None, "Colormap should not be None"
    
    # Test manual specification
    cmap_manual = adapter.get_obs_colormap(n_obs=16)
    assert cmap_manual is not None, "Manual colormap should not be None"
    
    print("✓ Colormap tests passed")


def analyze_results(observations, actions, path):
    """Analyze the generated sequence"""
    print("\nAnalyzing generated sequence...")
    
    # Observation statistics
    obs_unique = torch.unique(observations)
    print(f"Unique observations: {len(obs_unique)} out of 16 possible")
    print(f"Observation range: {observations.min().item()} - {observations.max().item()}")
    
    # Action statistics
    act_unique = torch.unique(actions)
    action_names = ["up", "down", "left", "right"]
    print(f"Actions used: {[action_names[i] for i in act_unique.tolist()]}")
    
    # Action distribution
    action_counts = torch.bincount(actions, minlength=4)
    for i, count in enumerate(action_counts):
        print(f"  {action_names[i]}: {count.item()} times ({count.item()/len(actions)*100:.1f}%)")
    
    # Path analysis
    positions = set(path)
    print(f"Unique positions visited: {len(positions)} out of {20*20} possible")
    
    # Coverage analysis
    coverage = len(positions) / (20 * 20) * 100
    print(f"Room coverage: {coverage:.1f}%")


def generate_long_sequence(seq_len=150000, save_path="."):
    """Generate and save a long sequence for CHMM training."""

    print(f"Generating long sequence of length {seq_len}...")
    adapter = RoomAdapter(room, seed=42)
    agent = Agent2D(seed=42)
    
    observations, actions, _ = agent.traverse(adapter, seq_len)
    
    # Save the sequences
    obs_path = os.path.join(save_path, "long_sequence_obs.pt")
    act_path = os.path.join(save_path, "long_sequence_act.pt")
    
    torch.save(observations, obs_path)
    torch.save(actions, act_path)
    
    print(f"✓ Long sequence saved to {obs_path} and {act_path}")
    return observations, actions

def main():
    """Run all tests and generate long sequence if needed."""
    import argparse
    parser = argparse.ArgumentParser(description="Run tests and generate data.")
    parser.add_argument("--generate-sequence", action="store_true", help="Generate and save a long sequence.")
    parser.add_argument("--seq-len", type=int, default=150000, help="Length of the sequence to generate.")
    args = parser.parse_args()

    if args.generate_sequence:
        generate_long_sequence(seq_len=args.seq_len)
    else:
        print("Running RoomAdapter and Agent2D tests...")
        try:
            # Run tests
            test_room_adapter_basic()
            test_device_compatibility()
            test_colormap()
            observations, actions, path = test_agent_traversal()
            
            # Analyze results
            analyze_results(observations, actions, path)
            
            print(" All tests passed successfully!")
            
            # Show first 10 steps
            print("First 10 steps:")
            for i in range(min(10, len(path))):
                print(f"Step {i}: pos={path[i]}, obs={observations[i].item()}, action={actions[i].item()}")
                
        except Exception as e:
            print(f"Test failed: {e}")
            raise

if __name__ == "__main__":
    main()
""