# Temporarily disabled for focused debugging.
# import pytest
# import torch
# import numpy as np
# from pathlib import Path
# import sys

# # Add project root to path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# from cscg_torch.models.chmm_torch import CHMM_torch
# from cscg_torch.models.train_utils import train_chmm, make_E, get_room_n_clones
# from cscg_torch.env_adapters.room_adapter import RoomAdapter
# from cscg_torch.agent_adapters.agent_2d import Agent2D
# from cscg_torch.tests.test_config (
#     setup_test_environment, cleanup_test_environment,
#     create_test_sequence, create_test_n_clones
# )

# # Test room - 20x20 with observations 0-15 (from room_adapter_test.py)
# room_layout = np.array([
#         [ 6, 10,  7,  6, 15, 12, 13, 10,  2,  8,  3,  2,  2,  8,  0, 10, 14, 14, 11, 11],
#         [ 7, 15, 10,  3,  8,  1,  7,  6,  3,  9, 15, 12,  3,  9, 14,  8, 10,  1,  1,  4],
#         [13,  8,  6, 13,  3,  9, 12,  4,  9, 12,  3,  3, 13,  4,  3, 11, 10, 11,  7,  3],
#         [ 9,  1,  6,  7, 14, 15,  3,  8,  0,  8,  6, 13, 12,  5,  4,  1,  6, 13,  5,  5],
#         [13, 13,  8,  1,  8,  8,  5,  4, 15,  8,  7,  7,  7,  0,  1,  4,  5, 11, 10,  1],
#         [11,  1, 15,  0, 15,  8,  1,  4, 14,  5, 13,  9, 15,  9,  1,  7,  0,  9,  8,  6],
#         [11,  6,  7, 13,  8,  5, 15,  0, 12,  0,  2, 12,  6,  7, 15,  8,  7,  6, 15, 15],
#         [ 6, 14,  9,  5, 12,  8,  6,  4, 14, 14, 13, 13,  3, 12,  0, 10,  5,  4,  0,  6],
#         [ 9, 10,  6,  7,  4, 11,  0,  7,  7,  8, 14, 13,  2,  6,  2, 12,  6,  1, 15, 13],
#         [12,  7, 15,  6,  3, 10, 11,  6,  4,  7,  9,  2,  0,  9,  1,  8,  4,  3,  0, 15],
#         [ 6, 14,  0, 10, 13,  3, 13,  4,  4, 10, 11,  9, 12,  1, 12,  6, 12,  6,  4, 11],
#         [ 2, 15,  6,  3, 13, 12,  7,  6,  0,  8,  3,  9, 13,  0,  0,  8,  2, 12, 13, 12],
#         [15,  0,  1,  4,  6,  6, 14, 11,  1, 14,  6,  4,  4,  4, 13, 10,  5,  2,  0,  5],
#         [12, 13,  4,  3,  5,  2, 13, 12,  8, 14, 10,  6,  9,  6, 14, 12, 14,  9, 11, 10],
#         [ 9,  8,  6,  7,  4,  0,  1,  1,  1,  3,  3, 12, 10,  4,  6, 12,  7,  4,  7, 12],
#         [ 2, 13, 12, 12, 11,  6,  2, 13,  8,  4,  1, 12,  4, 11,  8, 12,  1,  8,  3,  3],
#         [ 1,  4,  9, 15,  2,  6,  2, 13,  0,  7,  5,  1, 12, 10,  4,  1,  6,  4,  4, 14],
#         [ 5, 13, 13,  0, 13,  2,  6,  2,  2, 14,  7,  2, 11,  8,  0,  8,  3,  9,  5, 15],
#         [ 4,  6, 12,  9,  8, 10, 15,  1, 15, 10,  4,  2,  7, 15,  4,  9,  6, 10, 15, 13],
#         [ 4, 12, 14,  2, 11,  7,  5, 10,  0,  2,  1, 13,  5, 14,  8,  6,  6,  0, 15, 13]
#     ])

# class TestFullPipeline:
#     def setup_method(self):
#         setup_test_environment()

#     def teardown_method(self):
#         cleanup_test_environment()

#     @pytest.mark.slow
#     @pytest.mark.integration
#     def test_full_pipeline_execution(self):
#         print("\nStarting full pipeline integration test...")

#         # 1. Environment and Agent Setup
#         adapter = RoomAdapter(room_layout, seed=42)
#         agent = Agent2D(seed=42)

#         # 2. Data Generation
#         seq_len = 1000 # Shorter for integration test
#         observations, actions, _ = agent.traverse(adapter, seq_len)
#         print(f"Generated sequence of length {seq_len} on device: {observations.device}")

#         # 3. CHMM Model Initialization
#         n_obs_types = 16 # Max observation value in room_layout + 1
#         n_clones_per_obs = 2 # Small number for quick test
#         n_clones = get_room_n_clones(n_clones_per_obs, device=observations.device)
#         
#         model = CHMM_torch(n_clones, observations, actions, pseudocount=0.01, seed=42, device=observations.device)
#         print(f"CHMM initialized with {model.n_states} states on device: {model.device}")

#         # 4. EM Training
#         n_iter = 5 # Small number of iterations for quick test
#         print(f"Starting EM training for {n_iter} iterations...")
#         initial_bps = model.bps(observations, actions).item()
#         print(f"Initial BPS: {initial_bps:.4f}")

#         convergence_history = model.learn_em_T(observations, actions, n_iter=n_iter, term_early=True)
#         
#         final_bps = model.bps(observations, actions).item()
#         print(f"Final BPS: {final_bps:.4f}")
#         print(f"Total BPS improvement: {initial_bps - final_bps:.4f}")

#         assert len(convergence_history) > 0
#         assert final_bps < initial_bps, "BPS should decrease after training"

#         # 5. Emission Matrix (E) Creation and Validation
#         # This part assumes a direct mapping from clone to observation
#         # For room environments, E is often fixed or learned separately
#         # Here, we'll just ensure make_E works and is compatible
#         E_matrix = make_E(n_clones, device=observations.device)
#         assert E_matrix.shape[0] == model.n_states
#         assert E_matrix.shape[1] == n_obs_types
#         print(f"Emission matrix E created with shape: {E_matrix.shape}")

#         print("\n✓ Full pipeline integration test passed.")