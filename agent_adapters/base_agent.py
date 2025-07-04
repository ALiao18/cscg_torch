"""
Base Agent Adapter

Abstract base class for agent adapters used with environment adapters.
Provides common functionality for traversal and future reinforcement learning.
"""

import numpy as np
import torch
from typing import Tuple, List, Any
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Base class for all agent adapters.
    
    Provides common functionality for traversal and extensibility for 
    reinforcement learning capabilities.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize base agent.
        
        Args:
            seed: Random seed for reproducible behavior
        """
        assert isinstance(seed, int), f"seed must be int, got {type(seed)}"
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    @abstractmethod
    def traverse(self, environment: Any, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Traverse environment for specified sequence length.
        
        Args:
            environment: Environment to traverse
            seq_len: Number of steps to take
            
        Returns:
            Tuple of:
            - observations: torch.Tensor of shape (seq_len,) with dtype torch.int64
            - actions: torch.Tensor of shape (seq_len,) with dtype torch.int64  
            - path: List of coordinates showing traversed path
        """
        pass
    
    def reset_rng(self, seed: int = None):
        """
        Reset random number generator with new seed.
        
        Args:
            seed: New seed (uses original seed if None)
        """
        if seed is None:
            seed = self.seed
        self.rng = np.random.RandomState(seed)
    
    # Future methods for reinforcement learning can be added here
    # def learn(self, environment, episodes, **kwargs):
    #     """Learn policy through reinforcement learning"""
    #     pass
    # 
    # def act(self, state, **kwargs):
    #     """Select action based on learned policy"""
    #     pass
    # 
    # def update_policy(self, experience, **kwargs):
    #     """Update policy based on experience"""
    #     pass