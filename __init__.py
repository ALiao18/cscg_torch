"""
CSCG PyTorch Implementation

A GPU-optimized PyTorch implementation of Compositional Structured Context-Sensitive Grammar (CSCG)
Hidden Markov Models using the Baum-Welch EM algorithm.

This package provides:
- GPU-accelerated EM training for CHMM models
- Efficient forward-backward message passing
- Viterbi decoding and MAP inference
- Environment adapters for various domains
"""

from .models.chmm_torch import CHMM_torch
from .env_adapters import CSCGEnvironmentAdapter, RoomAdapter

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "CHMM_torch",
    "CSCGEnvironmentAdapter", 
    "RoomAdapter"
]