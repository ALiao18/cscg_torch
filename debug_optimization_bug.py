#!/usr/bin/env python3
"""
Debug why the optimization produces different results
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.chmm_torch import CHMM_torch
from models.train_utils import forward, backward, updateC as updateC_original

def debug_step_by_step():
    """Debug the updateC implementations step by step"""
    print("="*80)
    print("DEBUGGING updateC IMPLEMENTATION")
    print("="*80)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    obs_path = Path("long_sequence_obs.pt")
    act_path = Path("long_sequence_act.pt")
    
    if not obs_path.exists() or not act_path.exists():
        print("ERROR: Sequence files not found!")
        return
    
    x = torch.load(obs_path).to(device)
    a = torch.load(act_path).to(device)
    
    # Use very small sequence for detailed debugging
    seq_len = 10
    x_test = x[:seq_len]
    a_test = a[:seq_len]
    
    print(f"Debugging with {seq_len} steps:")
    print(f"x = {x_test}")
    print(f"a = {a_test}")
    
    # Setup model
    n_obs = 16
    n_clones_per_obs = 2  # Smaller for easier debugging
    n_clones = torch.full((n_obs,), n_clones_per_obs, dtype=torch.int64, device=device)
    
    model = CHMM_torch(n_clones, x_test, a_test, pseudocount=0.01, seed=42, device=device)
    
    # Prepare
    model._T_transposed = model.T.permute(0, 2, 1).contiguous()
    if not hasattr(model, '_workspace'):
        model._workspace = {}
    
    # Get messages
    log2_lik, mess_fwd = forward(
        model._T_transposed, model.Pi_x, model.n_clones,
        x_test, a_test, model.device, store_messages=True, workspace=model._workspace
    )
    mess_bwd = backward(model.T, model.n_clones, x_test, a_test, model.device, workspace=model._workspace)
    
    print(f"\nMessages computed:")
    print(f"mess_fwd shape: {mess_fwd.shape}")
    print(f"mess_bwd shape: {mess_bwd.shape}")
    
    # Debug the index computation manually
    print(f"\nDebugging index computation:")
    
    timesteps = len(x_test)
    state_loc = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones.cumsum(0)])
    mess_loc = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones[x_test].cumsum(0)])
    
    print(f"state_loc: {state_loc}")
    print(f"mess_loc: {mess_loc}")
    
    t_range = torch.arange(1, timesteps, device=device)
    i_indices_upd = x_test[:-1]
    j_indices_upd = x_test[1:]
    a_indices_upd = a_test[:-1]
    
    print(f"t_range: {t_range}")
    print(f"i_indices_upd: {i_indices_upd}")
    print(f"j_indices_upd: {j_indices_upd}")
    print(f"a_indices_upd: {a_indices_upd}")
    
    # Compute boundaries
    tm1_starts = mess_loc[t_range - 1]
    tm1_stops = mess_loc[t_range]
    t_starts = mess_loc[t_range]
    t_stops = mess_loc[t_range + 1]
    i_starts_upd = state_loc[i_indices_upd]
    i_stops_upd = state_loc[i_indices_upd + 1]
    j_starts_upd = state_loc[j_indices_upd]
    j_stops_upd = state_loc[j_indices_upd + 1]
    
    print(f"\nBoundaries:")
    print(f"tm1_starts: {tm1_starts}")
    print(f"tm1_stops: {tm1_stops}")
    print(f"t_starts: {t_starts}")
    print(f"t_stops: {t_stops}")
    print(f"i_starts_upd: {i_starts_upd}")
    print(f"i_stops_upd: {i_stops_upd}")
    print(f"j_starts_upd: {j_starts_upd}")
    print(f"j_stops_upd: {j_stops_upd}")
    
    # Manual step-by-step comparison
    print(f"\nStep-by-step comparison for first few steps:")
    
    # Compare .item() vs .tolist() approaches
    print(f"\nUsing .item() approach:")
    for idx in range(min(3, timesteps - 1)):
        ajt_item = a_indices_upd[idx].item()
        tm1_start_item = tm1_starts[idx].item()
        tm1_stop_item = tm1_stops[idx].item()
        
        print(f"  Step {idx}: ajt={ajt_item}, tm1_start={tm1_start_item}, tm1_stop={tm1_stop_item}")
    
    print(f"\nUsing .tolist() approach:")
    tm1_starts_list = tm1_starts.tolist()
    tm1_stops_list = tm1_stops.tolist()
    a_indices_list = a_indices_upd.tolist()
    
    for idx in range(min(3, timesteps - 1)):
        ajt_list = a_indices_list[idx]
        tm1_start_list = tm1_starts_list[idx]
        tm1_stop_list = tm1_stops_list[idx]
        
        print(f"  Step {idx}: ajt={ajt_list}, tm1_start={tm1_start_list}, tm1_stop={tm1_stop_list}")
    
    # Now test the actual computation for one step
    print(f"\nTesting computation for step 0:")
    
    idx = 0
    
    # Method 1: .item()
    ajt = a_indices_upd[idx].item()
    tm1_start = tm1_starts[idx].item()
    tm1_stop = tm1_stops[idx].item()
    t_start = t_starts[idx].item()
    t_stop = t_stops[idx].item()
    i_start = i_starts_upd[idx].item()
    i_stop = i_stops_upd[idx].item()
    j_start = j_starts_upd[idx].item()
    j_stop = j_stops_upd[idx].item()
    
    alpha1 = mess_fwd[tm1_start:tm1_stop]
    beta1 = mess_bwd[t_start:t_stop]
    T_slice1 = model.T[ajt, i_start:i_stop, j_start:j_stop]
    q1 = torch.outer(alpha1, beta1) * T_slice1
    
    print(f"Method 1 (.item()):")
    print(f"  alpha1: {alpha1}")
    print(f"  beta1: {beta1}")
    print(f"  T_slice1 shape: {T_slice1.shape}")
    print(f"  q1: {q1}")
    
    # Method 2: .tolist()
    ajt2 = a_indices_list[idx]
    tm1_start2 = tm1_starts_list[idx]
    tm1_stop2 = tm1_stops_list[idx]
    t_start2 = t_starts_list[idx]
    t_stop2 = t_stops_list[idx]
    i_start2 = i_starts_list[idx]
    i_stop2 = i_stops_list[idx]
    j_start2 = j_starts_list[idx]
    j_stop2 = j_stops_list[idx]
    
    alpha2 = mess_fwd[tm1_start2:tm1_stop2]
    beta2 = mess_bwd[t_start2:t_stop2]
    T_slice2 = model.T[ajt2, i_start2:i_stop2, j_start2:j_stop2]
    q2 = torch.outer(alpha2, beta2) * T_slice2
    
    print(f"Method 2 (.tolist()):")
    print(f"  alpha2: {alpha2}")
    print(f"  beta2: {beta2}")
    print(f"  T_slice2 shape: {T_slice2.shape}")
    print(f"  q2: {q2}")
    
    # Compare
    alpha_diff = torch.max(torch.abs(alpha1 - alpha2)).item()
    beta_diff = torch.max(torch.abs(beta1 - beta2)).item()
    q_diff = torch.max(torch.abs(q1 - q2)).item()
    
    print(f"\nDifferences:")
    print(f"  alpha diff: {alpha_diff}")
    print(f"  beta diff: {beta_diff}")
    print(f"  q diff: {q_diff}")
    
    if q_diff > 1e-10:
        print(f"  ❌ Found difference in computation!")
    else:
        print(f"  ✅ Computations are identical")

if __name__ == "__main__":
    debug_step_by_step()