"""
Comprehensive Tests for New Functions in CSCG_torch

Tests for newly added functions including:
- train_chmm function from train_utils.py
- plotting functions from base_adapter.py 
- room utilities from room_utils.py
- save_room_plot from room_adapter.py
"""

import torch
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cscg_torch.models.train_utils import (
    train_chmm, make_E, make_E_sparse, compute_forward_messages, place_field
)
from cscg_torch.env_adapters.room_utils import (
    get_obs_colormap, clone_to_obs_map, top_k_used_clones, count_used_clones,
    demo_room_setup
)
from cscg_torch.env_adapters.room_adapter import save_room_plot
from cscg_torch.env_adapters.base_adapter import plot_graph
from cscg_torch.models.chmm_torch import CHMM_torch


class TestTrainCHMM:
    """Test the train_chmm function."""
    
    def setup_method(self):
        """Set up test data."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_obs = 8
        self.n_clones = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2], device=self.device)
        self.seq_len = 100
        self.x = torch.randint(0, self.n_obs, (self.seq_len,), device=self.device)
        self.a = torch.randint(0, 4, (self.seq_len,), device=self.device)
    
    def test_train_chmm_basic(self):
        """Test basic train_chmm functionality."""
        trained_model, progression = train_chmm(
            self.n_clones, self.x, self.a, 
            device=self.device, n_iter=5, method='em_T'
        )
        
        # Validate model
        assert hasattr(trained_model, 'T'), "Model missing T matrix"
        assert hasattr(trained_model, 'Pi_x'), "Model missing Pi_x"
        assert trained_model.device == self.device, f"Device mismatch: {trained_model.device} != {self.device}"
        
        # Validate progression
        assert isinstance(progression, list), "Progression must be list"
        assert len(progression) <= 5, "Too many progression entries"
        assert all(isinstance(p, (int, float)) for p in progression), "Progression must contain numbers"
        
        # Validate T matrix normalization
        T_sums = trained_model.T.sum(dim=2)
        expected = torch.ones_like(T_sums)
        assert torch.allclose(T_sums, expected, atol=1e-5), "T matrix rows must sum to 1"
    
    def test_train_chmm_methods(self):
        """Test different training methods."""
        methods = ['em_T', 'viterbi_T']
        
        for method in methods:
            trained_model, progression = train_chmm(
                self.n_clones, self.x, self.a,
                device=self.device, n_iter=3, method=method
            )
            assert hasattr(trained_model, 'T'), f"Method {method} failed to produce T matrix"
            assert len(progression) <= 3, f"Method {method} exceeded iteration limit"
    
    def test_train_chmm_device_auto_detection(self):
        """Test automatic device detection."""
        trained_model, _ = train_chmm(
            self.n_clones, self.x, self.a, 
            device=None, n_iter=2  # Auto-detect device
        )
        
        assert trained_model.device.type in ['cuda', 'cpu'], "Invalid device detected"


class TestEmissionMatrixFunctions:
    """Test emission matrix creation functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_clones = torch.tensor([3, 2, 4], device=self.device)
    
    def test_make_E_basic(self):
        """Test basic dense emission matrix creation."""
        E = make_E(self.n_clones, self.device)
        
        # Shape validation
        assert E.shape == (9, 3), f"Wrong E shape: {E.shape}"
        assert E.device == self.device, f"Wrong device: {E.device}"
        
        # Normalization validation
        row_sums = E.sum(dim=1)
        expected = torch.ones(9, device=self.device)
        assert torch.allclose(row_sums, expected), "E rows must sum to 1"
        
        # Structure validation
        assert torch.all(E[0:3, 0] == 1.0), "First observation clones incorrect"
        assert torch.all(E[3:5, 1] == 1.0), "Second observation clones incorrect"
        assert torch.all(E[5:9, 2] == 1.0), "Third observation clones incorrect"
    
    def test_make_E_sparse(self):
        """Test sparse emission matrix creation."""
        E_sparse = make_E_sparse(self.n_clones, self.device)
        
        # Basic validation
        assert E_sparse.shape == (9, 3), f"Wrong sparse E shape: {E_sparse.shape}"
        assert E_sparse.is_sparse, "E should be sparse"
        assert E_sparse.device == self.device, f"Wrong device: {E_sparse.device}"
        
        # Compare with dense version
        E_dense = make_E(self.n_clones, self.device)
        E_sparse_dense = E_sparse.to_dense()
        assert torch.allclose(E_dense, E_sparse_dense), "Sparse and dense E matrices should match"
    
    def test_make_E_edge_cases(self):
        """Test edge cases for emission matrix creation."""
        # Single observation
        single_clone = torch.tensor([5], device=self.device)
        E_single = make_E(single_clone, self.device)
        assert E_single.shape == (5, 1), "Single observation E matrix wrong shape"
        
        # Large number of clones
        large_clones = torch.tensor([10, 20, 15], device=self.device)
        E_large = make_E(large_clones, self.device)
        assert E_large.shape == (45, 3), "Large E matrix wrong shape"


class TestForwardMessages:
    """Test forward message computation."""
    
    def setup_method(self):
        """Set up CHMM state for testing."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions, self.n_obs = 4, 6
        self.n_clones = torch.tensor([2, 2, 2, 2, 2, 2], device=self.device)
        self.n_states = self.n_clones.sum().item()
        
        # Create valid transition matrix
        T = torch.rand(self.n_actions, self.n_states, self.n_states, device=self.device)
        self.T = T / T.sum(dim=2, keepdim=True)
        
        # Create emission matrix
        self.E = make_E(self.n_clones, self.device)
        
        # Create initial distribution
        self.Pi_x = torch.ones(self.n_states, device=self.device) / self.n_states
        
        self.chmm_state = {
            'T': self.T,
            'E': self.E,
            'Pi_x': self.Pi_x,
            'n_clones': self.n_clones
        }
    
    def test_compute_forward_messages_basic(self):
        """Test basic forward message computation."""
        seq_len = 30
        x = torch.randint(0, self.n_obs, (seq_len,), device=self.device)
        a = torch.randint(0, self.n_actions, (seq_len,), device=self.device)
        
        mess_fwd = compute_forward_messages(self.chmm_state, x, a, self.device)
        
        # Shape validation
        assert mess_fwd.shape == (seq_len, self.n_states), f"Wrong forward message shape: {mess_fwd.shape}"
        assert mess_fwd.device == self.device, f"Wrong device: {mess_fwd.device}"
        
        # Probability validation
        assert torch.all(mess_fwd >= 0), "Forward messages must be non-negative"
        
        # Normalization validation (messages should be normalized)
        msg_sums = mess_fwd.sum(dim=1)
        expected = torch.ones(seq_len, device=self.device)
        assert torch.allclose(msg_sums, expected, atol=1e-5), "Forward messages must be normalized"
    
    def test_compute_forward_messages_gpu_compatibility(self):
        """Test GPU compatibility of forward message computation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        gpu_device = torch.device("cuda")
        
        # Move all data to GPU
        chmm_state_gpu = {key: val.to(gpu_device) for key, val in self.chmm_state.items()}
        x = torch.randint(0, self.n_obs, (20,), device=gpu_device)
        a = torch.randint(0, self.n_actions, (20,), device=gpu_device)
        
        mess_fwd = compute_forward_messages(chmm_state_gpu, x, a, gpu_device)
        assert mess_fwd.device == gpu_device, "GPU computation failed"


class TestPlaceField:
    """Test place field computation."""
    
    def setup_method(self):
        """Set up test data."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T, self.n_states = 20, 10
        self.mess_fwd = torch.rand(self.T, self.n_states, device=self.device)
        self.rc = torch.randint(0, 5, (self.T, 2), device=self.device)  # 5x5 grid
    
    def test_place_field_basic(self):
        """Test basic place field computation."""
        clone = 3
        field = place_field(self.mess_fwd, self.rc, clone, self.device)
        
        # Shape validation
        assert field.shape == (5, 5), f"Wrong field shape: {field.shape}"
        assert field.device == self.device, f"Wrong device: {field.device}"
        
        # Value validation
        assert torch.all(field >= 0), "Place field values must be non-negative"
        assert torch.isfinite(field).all(), "Place field values must be finite"
    
    def test_place_field_gpu_optimization(self):
        """Test GPU-optimized place field computation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        gpu_device = torch.device("cuda")
        mess_fwd_gpu = self.mess_fwd.to(gpu_device)
        rc_gpu = self.rc.to(gpu_device)
        
        field = place_field(mess_fwd_gpu, rc_gpu, 0, gpu_device)
        assert field.device == gpu_device, "GPU place field computation failed"
    
    def test_place_field_different_sizes(self):
        """Test place field computation with different grid sizes."""
        sizes = [(3, 3), (5, 5), (2, 8)]
        
        for max_r, max_c in sizes:
            # Create proper rc coordinates that actually span the expected range
            rc_r = torch.randint(0, max_r, (self.T, 1), device=self.device)
            rc_c = torch.randint(0, max_c, (self.T, 1), device=self.device)
            rc = torch.cat([rc_r, rc_c], dim=1)
            
            # Ensure we actually hit the maximum values
            rc[0, 0] = max_r - 1  # Ensure max row is visited
            rc[0, 1] = max_c - 1  # Ensure max col is visited
            
            field = place_field(self.mess_fwd, rc, 0, self.device)
            assert field.shape == (max_r, max_c), f"Wrong field shape for {max_r}x{max_c}: {field.shape}"


class TestRoomUtilities:
    """Test room utility functions."""
    
    def test_get_obs_colormap(self):
        """Test observation colormap creation."""
        n_obs_values = [4, 16, 20]
        
        for n_obs in n_obs_values:
            cmap = get_obs_colormap(n_obs)
            assert cmap is not None, f"Colormap creation failed for n_obs={n_obs}"
            # Test that we can call the colormap
            color = cmap(0)
            assert len(color) >= 3, "Colormap should return RGB values"
    
    def test_clone_to_obs_map(self):
        """Test clone to observation mapping."""
        n_clones = torch.tensor([3, 2, 4])
        
        mapping = clone_to_obs_map(n_clones)
        
        # Validate mapping structure
        assert len(mapping) == 9, f"Wrong mapping size: {len(mapping)}"
        assert all(isinstance(k, int) and isinstance(v, int) for k, v in mapping.items()), "Mapping must have int keys and values"
        
        # Validate mapping correctness
        assert mapping[0] == 0 and mapping[1] == 0 and mapping[2] == 0, "First observation mapping incorrect"
        assert mapping[3] == 1 and mapping[4] == 1, "Second observation mapping incorrect"
        assert mapping[5] == 2 and mapping[6] == 2 and mapping[7] == 2 and mapping[8] == 2, "Third observation mapping incorrect"
    
    def test_top_k_used_clones(self):
        """Test top k used clones function."""
        states = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
        
        top_clones = top_k_used_clones(states, k=3)
        
        # Validate structure
        assert len(top_clones) <= 3, "Too many top clones returned"
        assert all(isinstance(item, tuple) and len(item) == 2 for item in top_clones), "Top clones must be (clone, count) tuples"
        
        # Validate sorting (descending by count)
        if len(top_clones) > 1:
            assert top_clones[0][1] >= top_clones[1][1], "Top clones not sorted by frequency"
    
    def test_count_used_clones_mock(self):
        """Test count_used_clones with mock CHMM."""
        # Create a mock CHMM-like object
        class MockCHMM:
            def __init__(self):
                self.n_clones = torch.tensor([3, 2, 4])
            
            def decode(self, x, a):
                # Return mock states
                states = torch.tensor([0, 1, 2, 5, 6, 7, 8])
                return None, states
        
        mock_chmm = MockCHMM()
        x = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        a = torch.tensor([0, 1, 2, 3, 0, 1, 2])
        
        counts = count_used_clones(mock_chmm, x, a)
        
        # Validate structure
        assert isinstance(counts, dict), "Counts must be dict"
        assert len(counts) == 3, "Wrong number of observation types"
        assert all(isinstance(k, int) and isinstance(v, int) for k, v in counts.items()), "Counts must have int keys and values"
    
    def test_demo_room_setup(self):
        """Test demo room setup function."""
        adapter, n_clones, sample_data = demo_room_setup()
        
        # Validate adapter
        assert hasattr(adapter, 'reset'), "Adapter missing reset method"
        assert hasattr(adapter, 'step'), "Adapter missing step method"
        
        # Validate n_clones
        assert isinstance(n_clones, torch.Tensor), "n_clones must be tensor"
        assert n_clones.shape == (16,), "n_clones wrong shape"
        
        # Validate sample data
        x_seq, a_seq = sample_data
        assert isinstance(x_seq, np.ndarray), "x_seq must be numpy array"
        assert isinstance(a_seq, np.ndarray), "a_seq must be numpy array"
        assert len(x_seq) == len(a_seq), "Sequence length mismatch"


class TestPlottingFunctions:
    """Test plotting functions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create simple test room
        self.room = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  0,  1,  2, -1],
            [-1,  3,  4,  5, -1],
            [-1,  6,  7,  8, -1],
            [-1, -1, -1, -1, -1]
        ])
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_room_plot_basic(self):
        """Test basic room plot saving."""
        save_path = os.path.join(self.temp_dir, "test_room")
        
        save_room_plot(self.room, save_path)
        
        # Check that files were created
        assert os.path.exists(f"{save_path}.pdf"), "PDF file not created"
        assert os.path.exists(f"{save_path}.png"), "PNG file not created"
    
    def test_save_room_plot_formats(self):
        """Test different save formats."""
        save_path = os.path.join(self.temp_dir, "test_room_formats")
        
        save_room_plot(self.room, save_path, save_formats=['pdf', 'svg'])
        
        assert os.path.exists(f"{save_path}.pdf"), "PDF file not created"
        assert os.path.exists(f"{save_path}.svg"), "SVG file not created"
        assert not os.path.exists(f"{save_path}.png"), "PNG file should not be created"
    
    def test_save_room_plot_tensor_input(self):
        """Test room plot with tensor input."""
        room_tensor = torch.tensor(self.room)
        save_path = os.path.join(self.temp_dir, "test_room_tensor")
        
        save_room_plot(room_tensor, save_path)
        assert os.path.exists(f"{save_path}.pdf"), "PDF file not created for tensor input"


def test_integration_workflow():
    """Test complete integration workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create test environment
    adapter, n_clones, (x_seq, a_seq) = demo_room_setup()
    
    # Convert to tensors
    x = torch.tensor(x_seq, device=device)
    a = torch.tensor(a_seq, device=device)
    n_clones = n_clones.to(device)
    
    # 2. Train model
    trained_model, progression = train_chmm(
        n_clones, x, a, device=device, n_iter=3
    )
    
    # 3. Compute forward messages
    E = make_E(n_clones, device)
    chmm_state = {
        'T': trained_model.T,
        'E': E,
        'Pi_x': trained_model.Pi_x,
        'n_clones': n_clones
    }
    
    mess_fwd = compute_forward_messages(chmm_state, x, a, device)
    
    # 4. Test utility functions
    clone_map = clone_to_obs_map(n_clones)
    top_clones = top_k_used_clones(x.cpu().numpy(), k=5)
    
    # Validate complete workflow
    assert hasattr(trained_model, 'T'), "Training failed"
    assert mess_fwd.shape[0] == len(x), "Forward message computation failed"
    assert len(clone_map) == n_clones.sum().item(), "Clone mapping failed"
    assert len(top_clones) <= 5, "Top clones computation failed"
    
    print("✓ Complete integration workflow successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])