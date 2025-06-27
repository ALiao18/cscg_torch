"""
CSCG Torch Test Suite

Comprehensive testing framework for Clone-Structured Cognitive Graphs implementation.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

__version__ = "1.0.0"
__author__ = "CSCG Team"

# Test configuration constants
DEFAULT_SEQUENCE_LENGTH = 2000
DEFAULT_ROOM_SIZE = 5
DEFAULT_N_CLONES_PER_OBS = 2
DEFAULT_EM_ITERATIONS = 10

# Test data paths
TEST_ROOT = Path(__file__).parent
ROOMS_DIR = project_root / "rooms"
TEST_DATA_DIR = TEST_ROOT / "data"

# Ensure test data directory exists
TEST_DATA_DIR.mkdir(exist_ok=True)