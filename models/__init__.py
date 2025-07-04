"""
CSCG PyTorch Models

Core models and training utilities for CHMM with PyTorch.
"""

from .chmm_torch import CHMM_torch
from .train_utils import (
    validate_seq, forward, forwardE, forward_mp, forwardE_mp,
    backward, backwardE, updateC, updateCE, backtrace, backtraceE,
    forward_mp_all, backtrace_all, train_chmm, make_E, make_E_sparse,
    compute_forward_messages, compute_place_field
)

__all__ = [
    "CHMM_torch",
    "validate_seq",
    "forward", 
    "forwardE",
    "forward_mp",
    "forwardE_mp", 
    "backward",
    "backwardE",
    "updateC",
    "updateCE",
    "backtrace",
    "backtraceE",
    "forward_mp_all",
    "backtrace_all",
    "train_chmm",
    "make_E",
    "make_E_sparse", 
    "compute_forward_messages",
    "compute_place_field"
]