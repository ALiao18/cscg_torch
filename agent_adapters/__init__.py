"""
Agent Adapters Package

Contains agent classes for different environment types and RL algorithms.
"""

from .base_agent import BaseAgent
from .agent_2d import Agent2D

__all__ = ['BaseAgent', 'Agent2D']