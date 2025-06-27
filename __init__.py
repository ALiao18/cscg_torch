"""
CSCG PyTorch Implementation

A GPU-optimized PyTorch implementation of Compositional Structured Context-Sensitive Grammar (CSCG)
Hidden Markov Models using the Baum-Welch EM algorithm.

This package provides:
- GPU-accelerated EM training for CHMM models
- Efficient forward-backward message passing
- Viterbi decoding and MAP inference
- Environment adapters for various domains
- Plotting and visualization utilities
- Training utilities and helper functions
"""

from .models.chmm_torch import CHMM_torch
from .models.train_utils import (
    train_chmm, make_E, make_E_sparse, compute_forward_messages, place_field
)
from .env_adapters import (
    CSCGEnvironmentAdapter, RoomAdapter, plot_graph, save_room_plot,
    get_obs_colormap, clone_to_obs_map, top_k_used_clones, count_used_clones
)

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "CHMM_torch",
    "CSCGEnvironmentAdapter", 
    "RoomAdapter",
    "train_chmm",
    "make_E",
    "make_E_sparse",
    "compute_forward_messages", 
    "place_field",
    "plot_graph",
    "save_room_plot",
    "get_obs_colormap",
    "clone_to_obs_map",
    "top_k_used_clones",
    "count_used_clones"
]