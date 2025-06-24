"""
GPU-Optimized CHMM Implementation

Fully optimized for GPU computation with:
- Vectorized forward-backward algorithms
- Optimized memory access patterns
- Batched processing support
- Minimal CPU-GPU synchronization
"""

import torch
import numpy as np
from tqdm import trange
import sys

from .train_utils_optimized import (
    validate_seq_batch,
    forward_gpu_optimized, 
    backward_gpu_optimized,
    update_counts_gpu_optimized,
    forward_backward_vectorized,
    batched_em_step
)


class CHMM_torch_optimized:
    """
    GPU-Optimized Compositional Hidden Markov Model.
    
    Key optimizations:
    - Vectorized forward-backward algorithms
    - Proper GPU memory management
    - Batched sequence processing
    - Reduced CPU-GPU synchronization
    """
    
    def __init__(self, n_clones, x, a, pseudocount=0.0, dtype=torch.float32, seed=42):
        """
        Initialize optimized CHMM model.
        
        Args:
            n_clones (torch.Tensor): Number of clones per observation type
            x (torch.Tensor): Observation sequence(s)
            a (torch.Tensor): Action sequence(s)
            pseudocount (float): Pseudocount for regularization
            dtype (torch.dtype): Data type for parameters
            seed (int): Random seed
        """
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Setup device with optimal settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # Optimize CUDA settings for memory bandwidth
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Move and validate inputs
        self.n_clones = n_clones.to(self.device)
        x = x.to(self.device) 
        a = a.to(self.device)
        
        # Batch validation (minimal synchronization)
        validate_seq_batch(x, a, self.n_clones)
        
        # Model parameters
        self.dtype = dtype
        self.pseudocount = float(pseudocount)
        
        # Initialize model dimensions
        n_states = self.n_clones.sum().item()
        n_actions = a.max().item() + 1
        
        # Pre-allocate parameter tensors with optimal memory layout
        self.C = torch.zeros(n_actions, n_states, n_states, 
                            dtype=dtype, device=self.device)
        self.Pi_x = torch.ones(n_states, dtype=dtype, device=self.device) / n_states
        self.Pi_a = torch.ones(n_actions, dtype=dtype, device=self.device) / n_actions
        
        # Initialize with small random values for better optimization
        torch.nn.init.uniform_(self.C, 0.1, 1.0)
        
        # Compute initial transition matrix
        self.update_T()
        
        print(f"🚀 Optimized CHMM initialized on {self.device}")
        print(f"   States: {n_states}, Actions: {n_actions}")
        print(f"   Memory optimizations: {'available' if torch.cuda.is_available() else 'not available'}")
    
    def update_T(self, verbose=False):
        """Update transition matrix with GPU-optimized operations."""
        # Vectorized normalization with numerical stability
        self.T = self.C + self.pseudocount
        
        # Efficient normalization along last dimension
        norm = self.T.sum(dim=2, keepdim=True)
        norm = torch.clamp(norm, min=1e-10)  # Avoid division by zero
        self.T = self.T / norm
        
        if verbose:
            print(f"   T shape: {self.T.shape}")
            print(f"   T device: {self.T.device}")
            # Minimal validation - avoid .sum() which causes sync
            print(f"   T memory: {self.T.numel() * 4 / 1e6:.1f}MB")
    
    def bps_optimized(self, x, a, reduce=True):
        """
        Compute bits-per-step with GPU optimization.
        
        Args:
            x (torch.Tensor): Observation sequence(s)
            a (torch.Tensor): Action sequence(s)
            reduce (bool): Return total or per-step values
            
        Returns:
            torch.Tensor: Negative log2-likelihood
        """
        # Ensure sequences are on device
        x, a = x.to(self.device), a.to(self.device)
        
        # GPU-optimized forward pass
        T_tr = self.T.permute(0, 2, 1)
        log2_lik, _ = forward_gpu_optimized(
            T_tr, self.Pi_x, self.n_clones, x, a, self.device, store_messages=True
        )
        
        return -log2_lik.sum() if reduce else -log2_lik
    
    def learn_em_optimized(self, x, a, n_iter=100, term_early=True, batch_size=None):
        """
        GPU-optimized EM training with batched processing.
        
        Args:
            x (torch.Tensor): Observation sequence(s)
            a (torch.Tensor): Action sequence(s)
            n_iter (int): Maximum EM iterations
            term_early (bool): Early termination on convergence
            batch_size (int): Batch size for multiple sequences
            
        Returns:
            list: Convergence history (bits per step)
        """
        print(f"🚀 Starting optimized EM training...")
        sys.stdout.flush()
        
        # Move sequences to device
        x, a = x.to(self.device), a.to(self.device)
        
        # Setup for batched or single sequence processing
        if batch_size is not None and x.ndim == 1:
            # Split single long sequence into batches
            seq_len = len(x)
            n_batches = seq_len // batch_size
            x_batched = x[:n_batches * batch_size].view(n_batches, batch_size)
            a_batched = a[:n_batches * batch_size].view(n_batches, batch_size)
            use_batching = True
        else:
            x_batched = x
            a_batched = a
            use_batching = False
        
        convergence = []
        pbar = trange(n_iter, desc="EM Optimization")
        
        # Optimization: pre-allocate frequently used tensors
        prev_bps = float('inf')
        
        for iteration in pbar:
            # === E-step: Compute expected sufficient statistics ===
            if use_batching:
                total_log_lik, C_new = batched_em_step(
                    self.T, self.Pi_x, self.n_clones, x_batched, a_batched, self.device
                )
                self.C.copy_(C_new)
            else:
                # Single sequence processing with optimized algorithms
                log2_lik, mess_fwd, mess_bwd = forward_backward_vectorized(
                    self.T, self.Pi_x, self.n_clones, x, a, self.device
                )
                
                # Update counts
                update_counts_gpu_optimized(
                    self.C, self.T, self.n_clones, mess_fwd, mess_bwd, x, a, self.device
                )
                
                total_log_lik = log2_lik.sum()
            
            # === M-step: Update parameters ===
            self.update_T(verbose=False)
            
            # === Convergence tracking (minimal CPU-GPU sync) ===
            current_bps = (-total_log_lik / len(x)).item()
            convergence.append(current_bps)
            
            # Update progress bar
            pbar.set_postfix({
                'BPS': f'{current_bps:.4f}',
                'Δ': f'{prev_bps - current_bps:+.6f}'
            })
            
            # Early termination check
            if term_early and current_bps >= prev_bps:
                print(f"\n✅ Converged at iteration {iteration}")
                break
            
            prev_bps = current_bps
            
            # Memory cleanup every 10 iterations
            if iteration % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"✅ EM training completed: {len(convergence)} iterations")
        print(f"   Final BPS: {convergence[-1]:.4f}")
        print(f"   Total improvement: {convergence[0] - convergence[-1]:.4f}")
        
        return convergence
    
    def decode_optimized(self, x, a):
        """
        GPU-optimized Viterbi decoding.
        
        Args:
            x (torch.Tensor): Observation sequence
            a (torch.Tensor): Action sequence
            
        Returns:
            tuple: (log_likelihood, most_likely_states)
        """
        x, a = x.to(self.device), a.to(self.device)
        
        # Use max-product forward pass for Viterbi
        T_tr = self.T.permute(0, 2, 1)
        
        # Simplified Viterbi - could be further optimized with proper backtrace
        log2_lik, mess_fwd = forward_gpu_optimized(
            T_tr, self.Pi_x, self.n_clones, x, a, self.device, store_messages=True
        )
        
        # Find most likely states (simplified - proper Viterbi needs backtrace)
        # This is a placeholder for the full Viterbi implementation
        if mess_fwd.ndim == 2:
            states = mess_fwd.argmax(dim=1)
        else:
            states = torch.zeros(len(x), dtype=torch.int64, device=self.device)
        
        return -log2_lik.sum(), states
    
    def get_memory_usage(self):
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        else:
            return "CPU mode - no GPU memory tracking"
    
    def benchmark_speedup(self, x, a, n_iter=5):
        """
        Benchmark the speedup of optimized vs standard implementation.
        
        Returns performance metrics.
        """
        print(f"🔬 Benchmarking optimized implementation...")
        
        x, a = x.to(self.device), a.to(self.device)
        
        # Warm up GPU
        for _ in range(3):
            _ = self.bps_optimized(x, a)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark optimized version
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        else:
            import time
            start = time.time()
        
        for _ in range(n_iter):
            convergence = self.learn_em_optimized(x, a, n_iter=3, term_early=False)
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            elapsed = time.time() - start
        
        avg_time_per_iter = elapsed / (n_iter * 3)  # 3 EM iterations per call
        
        print(f"⚡ Performance Results:")
        print(f"   Average time per EM iteration: {avg_time_per_iter:.4f}s")
        print(f"   Sequence length: {len(x)}")
        print(f"   Processing rate: {len(x) / avg_time_per_iter:.0f} steps/second")
        print(f"   Memory usage: {self.get_memory_usage()}")
        
        return {
            'avg_time_per_iter': avg_time_per_iter,
            'steps_per_second': len(x) / avg_time_per_iter,
            'final_bps': convergence[-1] if convergence else None
        }