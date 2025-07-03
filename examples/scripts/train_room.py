#!/usr/bin/env python3
"""
Configurable CSCG training script for 20x20 room
"""
import torch
import numpy as np
import argparse
from tqdm import trange

from models.train_utils import train_chmm
from env_adapters.room_utils import create_room_adapter, get_room_n_clones, generate_room_sequence

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CSCG model on room navigation task')
    
    # Training parameters
    parser.add_argument('--seq_len', type=int, default=150000,
                       help='Length of training sequence (default: 15000)')
    parser.add_argument('--n_clones_per_obs', type=int, default=150,
                       help='Number of clones per observation type (default: 150)')
    parser.add_argument('--n_iter', type=int, default=1,
                       help='Number of EM iterations (default: 100)')
    parser.add_argument('--pseudocount', type=float, default=0.01,
                       help='Pseudocount for smoothing (default: 0.01)')
    
    # Training method
    parser.add_argument('--method', type=str, default='em_T', 
                       choices=['em_T', 'viterbi_T', 'em_E'],
                       help='Training method (default: em_T)')
    parser.add_argument('--learn_E', action='store_true',
                       help='Also learn emission matrix E')
    parser.add_argument('--viterbi', action='store_true',
                       help='Use Viterbi (hard) EM instead of soft EM')
    parser.add_argument('--no_early_stopping', action='store_true',
                       help='Disable early stopping')
    
    # Device and performance
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Output options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--test_device', action='store_true',
                       help='Run device performance test first')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples and exit')
    
    return parser.parse_args()

def select_device(device_arg):
    """Select compute device based on argument and availability"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            raise RuntimeError("CUDA requested but not available")
    elif device_arg == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device("mps:0")
        else:
            raise RuntimeError("MPS requested but not available")
    else:  # cpu
        device = torch.device("cpu")
    
    return device

def test_device_performance(device):
    """Quick device performance test"""
    print(f"\n=== Device Performance Test ===")
    try:
        test_tensor = torch.randn(1000, 1000, device=device)
        result = test_tensor @ test_tensor.T
        print(f"Matrix multiplication successful on {result.device}")
        print(f"Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"Device test failed: {e}")
        return False

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Show examples if requested
    if args.examples:
        show_help()
        return None, None, None
    
    # Setup device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Optional device test
    if args.test_device:
        if not test_device_performance(device):
            print("Device test failed, exiting...")
            return
    
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
    
    # Print training configuration
    print(f"\n=== Training Configuration ===")
    print(f"Sequence length: {args.seq_len}")
    print(f"Clones per observation: {args.n_clones_per_obs}")
    print(f"EM iterations: {args.n_iter}")
    print(f"Training method: {args.method}")
    print(f"Learn emissions: {args.learn_E}")
    print(f"Use Viterbi: {args.viterbi}")
    print(f"Early stopping: {not args.no_early_stopping}")
    print(f"Pseudocount: {args.pseudocount}")
    print(f"Random seed: {args.seed}")
    
    # Convert room data to tensor on correct device
    room_tensor = torch.tensor(room_data, device=device, dtype=torch.int64)
    adapter = create_room_adapter(room_tensor, adapter_type="torch", seed=args.seed)
    print(f"Room shape: {room_data.shape}")
    
    # Generate training sequence
    print(f"\nGenerating sequence of length {args.seq_len}...")
    x_seq, a_seq = generate_room_sequence(adapter, args.seq_len, device=device)
    
    # Convert to tensors
    x = torch.tensor(x_seq, device=device, dtype=torch.int64)
    a = torch.tensor(a_seq, device=device, dtype=torch.int64)
    
    # Setup model parameters
    n_clones = get_room_n_clones(n_clones_per_obs=args.n_clones_per_obs, device=device)
    total_states = n_clones.sum().item()
    print(f"Training with {len(n_clones)} observation types and {total_states} total clones")
    
    if args.verbose:
        print(f"Training tensors on device: x={x.device}, a={a.device}")
    
    # Train model
    print(f"\nTraining CSCG model...")
    model, progression = train_chmm(
        n_clones, x, a,
        device=device,
        method=args.method,
        n_iter=args.n_iter,
        learn_E=args.learn_E,
        viterbi=args.viterbi,
        pseudocount=args.pseudocount,
        early_stopping=(not args.no_early_stopping),
        seed=args.seed
        )
    
    # Evaluate
    print(f"\n=== Final Evaluation ===")
    final_bps = model.bps(x, a, reduce=True)
    print(f"Final bits per symbol: {final_bps:.4f}")
    
    if len(progression) > 1:
        improvement = progression[0] - progression[-1]
        print(f"Total improvement: {improvement:.4f} BPS")
        print(f"Improvement percentage: {100 * improvement / progression[0]:.2f}%")
    
    # Decode states
    if args.verbose:
        print("Decoding optimal state sequence...")
        _, states = model.decode(x, a)
        print(f"Decoded {len(states)} states")
    
    print(f"\nTraining completed successfully!")
    print(f"Final model device: {model.device}")
    
    return model, progression, final_bps

def show_help():
    """Show usage examples"""
    print("Usage examples:")
    print("  # Quick test with small parameters")
    print("  python train_room.py --seq_len 1000 --n_clones_per_obs 10 --n_iter 5")
    print("")
    print("  # GPU-accelerated long sequence training")
    print("  python train_room.py --seq_len 100000 --n_clones_per_obs 200 --learn_E")
    print("")
    print("  # Force CPU training")
    print("  python train_room.py --device cpu --verbose")
    print("")
    print("  # Viterbi training with custom parameters")
    print("  python train_room.py --method viterbi_T --pseudocount 0.1 --no_early_stopping")
    print("")
    print("  # Large scale training with GPU acceleration")
    print("  python train_room.py --seq_len 500000 --n_clones_per_obs 500 --n_iter 200")

if __name__ == "__main__":
    try:
        model, progression, bps = main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()