import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import torch
from tqdm import tqdm
import pytest
from cscg_torch.models.chmm_torch import CHMM_torch
from cscg_torch.tests.test_config import setup_test_environment, cleanup_test_environment, create_test_sequence, create_test_n_clones

class TestCHMMTorchBasic:
    def setup_method(self):
        setup_test_environment()

    def teardown_method(self):
        cleanup_test_environment()

    def test_basic_initialization(self):
        x, a = create_test_sequence()
        n_clones = create_test_n_clones()
        model = CHMM_torch(n_clones, x, a)
        assert hasattr(model, 'T')
        assert hasattr(model, 'Pi_x')
        print("\n✓ Basic CHMM_torch initialization test passed.")

    def test_bps_computation(self):
        x, a = create_test_sequence()
        n_clones = create_test_n_clones()
        model = CHMM_torch(n_clones, x, a)
        
        bps_reduced = model.bps(x, a, reduce=True)
        assert isinstance(bps_reduced, torch.Tensor)
        assert bps_reduced.ndim == 0
        assert torch.isfinite(bps_reduced)
        
        print("\n✓ BPS computation test passed.")

    def test_em_training(self):
        x, a = create_test_sequence()
        n_clones = create_test_n_clones()
        model = CHMM_torch(n_clones, x, a)
        
        initial_bps = model.bps(x, a, reduce=True).item()
        convergence_history = model.learn_em_T(x, a, n_iter=5)
        final_bps = model.bps(x, a, reduce=True).item()
        
        assert isinstance(convergence_history, list)
        assert len(convergence_history) > 0
        assert all(isinstance(val, (float, int)) for val in convergence_history)
        assert final_bps < initial_bps, "EM training did not improve BPS"
        
        print("\n✓ EM training test passed.")

    @pytest.mark.longrun
    def test_long_sequence_training(self):
        print("\nStarting long sequence training test...")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        # Define paths for generated sequences
        obs_path = Path("long_sequence_obs.pt")
        act_path = Path("long_sequence_act.pt")

        # Check if sequence files exist, if not, instruct user to generate
        if not obs_path.exists() or not act_path.exists():
            print("\n" + "="*80)
            print("WARNING: Long sequence files not found.")
            print("Please generate them by running the following command:")
            print("python /Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze/cscg_torch/tests/env_adapters/room_adapter_test.py --generate-sequence")
            print("Then re-run this test.")
            print("="*80 + "\n")
            pytest.skip("Long sequence files not found, skipping test.")

        print("Loading long sequence data...")
        x = torch.load(obs_path).to(device)
        a = torch.load(act_path).to(device)

        # Model parameters
        n_obs = 16 # Based on the room_adapter_test.py room
        n_clones_per_obs = 10 # Temporarily reduced for faster debugging
        n_clones = torch.full((n_obs,), n_clones_per_obs, dtype=torch.int64, device=device)

        # Initialize model
        model = CHMM_torch(n_clones, x[:1000], a[:1000], pseudocount=0.01, seed=42, device=device) # Temporarily reduced sequence length

        # Training
        n_iter = 2 # Temporarily reduced iterations for faster ging
        print(f"Starting EM training for {n_iter} iterations...")
        initial_bps = model.bps(x, a).item()
        print(f"Initial BPS: {initial_bps:.4f}")

        pbar = tqdm(range(n_iter), desc="EM Training Progress")
        for i in pbar:
            # print(f"\n--- Iteration {i+1}/{n_iter} ---") # Removed as tqdm handles iteration display
            convergence = model.learn_em_T(x, a, n_iter=1, term_early=False)
            current_bps = convergence[0]
            pbar.set_postfix(bps=f'{current_bps:.4f}')
            # print(f"BPS after iteration {i+1}: {current_bps:.4f}") # Removed as tqdm handles postfix

        final_bps = model.bps(x, a).item()
        print(f"\nFinal BPS after {n_iter} iterations: {final_bps:.4f}")
        print(f"Total improvement: {initial_bps - final_bps:.4f}")

        print("\n✓ Long sequence training test passed.")

if __name__ == "__main__":
    pytest.main([__file__])

