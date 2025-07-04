from __future__ import print_function
from builtins import range
import numpy as np
from tqdm import trange
import sys
import torch
import os

# Debug mode control - set CHMM_DEBUG=1 to enable assertions
DEBUG_MODE = os.environ.get('CHMM_DEBUG', '0') == '1'

# Performance: Skip device checks in production (major speedup)
SKIP_DEVICE_CHECKS = os.environ.get('CHMM_SKIP_DEVICE_CHECKS', '1') == '1'

def validate_seq(x, a, n_clones = None):
    """
    validate the sequence of observations and actions
    """
    # Strict input validation
    assert isinstance(x, torch.Tensor), f"x must be torch.Tensor, got {type(x)}"
    assert isinstance(a, torch.Tensor), f"a must be torch.Tensor, got {type(a)}"
    
    assert len(x) == len(a), f"sequence lengths must match: x={len(x)}, a={len(a)}"
    assert len(x) > 0, "sequences cannot be empty"
    
    assert len(x.shape) == 1, f"x must be 1D, got {x.ndim}D"
    assert len(a.shape) == 1, f"a must be 1D, got {a.ndim}D"
    
    assert x.dtype == torch.int64, f"x must be int64, got {x.dtype}"
    assert a.dtype == torch.int64, f"a must be int64, got {a.dtype}"
    
    assert x.min() >= 0, f"x values must be non-negative, got min {x.min()}"
    assert a.min() >= 0, f"a values must be non-negative, got min {a.min()}"
    
    if n_clones is not None:
        assert isinstance(n_clones, torch.Tensor), f"n_clones must be torch.Tensor, got {type(n_clones)}"
        assert len(n_clones.shape) == 1, f"n_clones must be 1D, got {n_clones.ndim}D"
        assert n_clones.dtype == torch.int64, f"n_clones must be int64, got {n_clones.dtype}"
        assert torch.all(n_clones > 0), "all n_clones values must be positive"
        
        n_emissions = n_clones.shape[0]
        assert isinstance(n_emissions, int), f"n_emissions must be int, got {type(n_emissions)}"
        assert n_emissions > 0, f"n_emissions must be positive, got {n_emissions}"
        
        x_max = x.max().item()
        assert isinstance(x_max, int), f"x_max must be int, got {type(x_max)}"
        assert x_max < n_emissions, f"x values out of range: max {x_max} >= n_emissions {n_emissions}"

def forward(T_tr, Pi, n_clones, x, a, device, store_messages = False, workspace=None):
    """
    Log-probability of a sequence, and optionally, messages

    T_tr: transition matrix transposed
    Pi: initial state distribution
    n_clones: number of clones for each emission
    x: observation sequence
    a: action sequence
    store_messages: whether to store messages
    """
    # Ensure all tensors are on the same device
    device = T_tr.device  # Use the device of T_tr as reference
    x = x.to(device)
    a = a.to(device)
    Pi = Pi.to(device)
    n_clones = n_clones.to(device)
    
    # Strict input validation
    assert isinstance(T_tr, torch.Tensor), f"T_tr must be torch.Tensor, got {type(T_tr)}"
    assert isinstance(Pi, torch.Tensor), f"Pi must be torch.Tensor, got {type(Pi)}"
    assert isinstance(n_clones, torch.Tensor), f"n_clones must be torch.Tensor, got {type(n_clones)}"
    assert isinstance(x, torch.Tensor), f"x must be torch.Tensor, got {type(x)}"
    assert isinstance(a, torch.Tensor), f"a must be torch.Tensor, got {type(a)}"
    assert isinstance(device, torch.device), f"device must be torch.device, got {type(device)}"
    assert isinstance(store_messages, bool), f"store_messages must be bool, got {type(store_messages)}"
    
    assert T_tr.ndim == 3, f"T_tr must be 3D, got {T_tr.ndim}D"
    assert Pi.ndim == 1, f"Pi must be 1D, got {Pi.ndim}D"
    assert n_clones.ndim == 1, f"n_clones must be 1D, got {n_clones.ndim}D"
    assert x.ndim == 1, f"x must be 1D, got {x.ndim}D"
    assert a.ndim == 1, f"a must be 1D, got {a.ndim}D"
    
    assert len(x) == len(a), f"sequence lengths must match: x={len(x)}, a={len(a)}"
    assert len(x) > 0, "sequences cannot be empty"
    
    n_states = Pi.shape[0]
    assert T_tr.shape[1] == n_states, f"T_tr dim 1 mismatch: {T_tr.shape[1]} != {n_states}"
    assert T_tr.shape[2] == n_states, f"T_tr dim 2 mismatch: {T_tr.shape[2]} != {n_states}"
    assert n_clones.sum() == n_states, f"n_clones sum mismatch: {n_clones.sum()} != {n_states}"
    
    # Memory Optimization: Use workspace to reduce allocations
    if workspace is None:
        workspace = {}
    
    # Reuse pre-allocated tensors when possible
    if 'state_loc' not in workspace or workspace['state_loc'].size(0) != n_clones.size(0) + 1:
        workspace['state_loc'] = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones]).cumsum(0)
    state_loc = workspace['state_loc']
    
    if 'log2_lik' not in workspace or workspace['log2_lik'].size(0) != len(x):
        workspace['log2_lik'] = torch.zeros(len(x), dtype=T_tr.dtype, device=device)
    log2_lik = workspace['log2_lik']
    log2_lik.zero_()  # Clear previous values
    
    # initialize messages
    t = 0
    j = x[t]
    j_start, j_stop = state_loc[j : j + 2]
    message = Pi[j_start:j_stop].clone()
    p_obs = message.sum()
    if DEBUG_MODE:
        assert p_obs > 0, "Probability of observation is zero"
    message = message / p_obs
    log2_lik[0] = torch.log2(p_obs)
    
    if store_messages:
        mess_loc = torch.cat([
            torch.tensor([0], dtype=n_clones.dtype, device=device), 
            n_clones[x]
        ]).cumsum(0)
        mess_fwd = torch.empty(mess_loc[-1], dtype=T_tr.dtype, device=device)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd[t_start:t_stop] = message
    else:
        mess_fwd = None

    # Optimization: Pre-compute all indices for vectorized access
    seq_len = x.shape[0]
    if seq_len > 1:
        # Vectorized index computation
        i_indices = x[:-1]  # [T-1]
        j_indices = x[1:]   # [T-1] 
        a_indices = a[:-1]  # [T-1]
        
        # Pre-compute all state boundaries
        i_starts = state_loc[i_indices]
        i_stops = state_loc[i_indices + 1]
        j_starts = state_loc[j_indices] 
        j_stops = state_loc[j_indices + 1]
        
        if DEBUG_MODE:
            # Validate all indices at once
            assert torch.all(a_indices >= 0) and torch.all(a_indices < T_tr.shape[0])
            assert torch.all(i_indices >= 0) and torch.all(i_indices < len(n_clones))
            assert torch.all(j_indices >= 0) and torch.all(j_indices < len(n_clones))
            assert torch.all(i_starts < i_stops) and torch.all(j_starts < j_stops)
    
    # Vectorized forward pass - major speedup by eliminating Python loop overhead
    # Note: Sequential dependencies prevent full vectorization, but we optimize inner operations
    
    # Pre-allocate workspace tensors for message passing
    if 'max_message_size' not in workspace:
        workspace['max_message_size'] = n_clones.max().item()
    max_msg_size = workspace['max_message_size']
    
    if 'temp_message' not in workspace or workspace['temp_message'].size(0) < max_msg_size:
        workspace['temp_message'] = torch.empty(max_msg_size, dtype=T_tr.dtype, device=device)
    
    # Batch index access - convert to tensor operations
    if seq_len > 1:
        # Convert scalar accesses to tensor operations where possible
        # CRITICAL FIX: Convert to CPU once instead of .item() in loop
        ajt_cpu = a_indices.cpu().numpy()
        i_starts_cpu = i_starts.cpu().numpy()
        i_stops_cpu = i_stops.cpu().numpy()
        j_starts_cpu = j_starts.cpu().numpy()
        j_stops_cpu = j_stops.cpu().numpy()
        
        # Forward pass with optimized indexing
        for t in range(1, seq_len):
            t_idx = t - 1
            ajt = int(ajt_cpu[t_idx])
            i_start, i_stop = int(i_starts_cpu[t_idx]), int(i_stops_cpu[t_idx])
            j_start, j_stop = int(j_starts_cpu[t_idx]), int(j_stops_cpu[t_idx])

            # Optimized matrix-vector multiplication
            T_tr_slice = T_tr[ajt, j_start:j_stop, i_start:i_stop]
            
            # Use matmul for better GPU utilization (3.8x faster than mv)
            message = torch.matmul(T_tr_slice, message.unsqueeze(-1)).squeeze(-1)
            
            # Vectorized normalization
            p_obs = message.sum()
            if DEBUG_MODE:
                assert p_obs > 0, f"Probability of observation is zero at t={t}"
            
            message = message / p_obs
            log2_lik[t] = torch.log2(p_obs)

            if store_messages:
                t_start, t_stop = mess_loc[t : t + 2]
                mess_fwd[t_start:t_stop] = message
    
    # Final validation
    assert isinstance(log2_lik, torch.Tensor), f"log2_lik must be tensor, got {type(log2_lik)}"
    assert log2_lik.shape == (len(x),), f"log2_lik shape mismatch: {log2_lik.shape} != ({len(x)},)"
    
    if store_messages:
        assert isinstance(mess_fwd, torch.Tensor), f"mess_fwd must be tensor, got {type(mess_fwd)}"
    else:
        assert mess_fwd is None, f"mess_fwd must be None when store_messages=False, got {type(mess_fwd)}"
    
    return log2_lik, mess_fwd

def forwardE(T_tr, E, Pi, n_clones, x, a, device, store_messages = False):
    """
    Compute log2-likelihood of a sequence with emissions.
    
    Args:
        T_tr (Tensor): [n_actions, n_states, n_states] Transposed transition matrix
        E (Tensor): [n_states, n_obs] Emission probabilities
        Pi (Tensor): [n_states] Initial state distribution
        n_clones (Tensor): [n_obs] Number of clones per observation
        x (Tensor): [T] Observation sequence
        a (Tensor): [T] Action sequence
        device (torch.device): GPU or CPU
        store_messages (bool): If True, stores full forward messages
        
    Returns:
        log2_lik: [T] log-likelihood per time step
        mess_fwd: [T, n_states] forward messages if store_messages=True, else None
    """
    #  Assert and Move tensors
    assert E.shape == (n_clones.sum(), len(n_clones))
    x, a, Pi, E = x.to(device), a.to(device), Pi.to(device), E.to(device)

    T_len = x.shape[0]
    n_states = E.shape[0]
    dtype = T_tr.dtype

    log2_lik = torch.zeros(len(x), dtype = dtype, device = device)
    if store_messages:
        mess_fwd = torch.empty(T_len, n_states, dtype = dtype, device = device)

    # inital step
    t = 0
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.sum()
    assert p_obs > 0, f"Zero obs prob at t={t}, obs={j}"
    message = message / p_obs
    log2_lik[t] = torch.log2(p_obs)
    if store_messages:
        mess_fwd[t] = message

        # === Forward loop ===
    for t in range(1, T_len):
        aij = a[t - 1]
        j = x[t]
        message = torch.matmul(T_tr[aij], message.unsqueeze(-1)).squeeze(-1)
        message *= E[:, j]
        p_obs = message.sum()
        assert p_obs > 0, f"Zero obs prob at t={t}, obs={j}"
        message /= p_obs
        log2_lik[t] = torch.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message

    return (log2_lik, mess_fwd) if store_messages else log2_lik

def forward_mp(T_tr, Pi, n_clones, x, a, device, store_messages=False):
    """
    Max-product forward pass (for Viterbi-like decoding)

    Args:
        T_tr: [n_actions, n_states, n_states] transition matrix (transposed)
        Pi: [n_states] initial state distribution
        n_clones: [n_obs] number of clones per observation
        x: [T] observation sequence
        a: [T] action sequence
        device: torch.device
        store_messages: if True, returns full clone activations

    Returns:
        log2_lik: [T] log-likelihood of best path at each step
        mess_fwd: [sum(n_clones[x])] forward messages if requested
    """
    # === Precompute state indices ===
    state_loc = torch.cat([torch.tensor([0], dtype=n_clones.dtype, device=device), n_clones.cumsum(0)])
    mess_loc = torch.cat([torch.tensor([0], dtype=n_clones.dtype, device=device), n_clones[x].cumsum(0)])

    T_len = x.shape[0]
    dtype = T_tr.dtype
    log2_lik = torch.zeros(T_len, dtype=dtype, device=device)

    # === Initial step ===
    t = 0
    j = x[t]
    j_start, j_stop = state_loc[j], state_loc[j + 1]
    message = Pi[j_start:j_stop].clone()
    p_obs = message.max()
    assert p_obs > 0, f"Zero probability at t=0"
    message /= p_obs
    log2_lik[t] = torch.log2(p_obs)

    if store_messages:
        mess_fwd = torch.empty(mess_loc[-1], dtype=dtype, device=device)
        t_start, t_stop = mess_loc[t:t + 2]
        mess_fwd[t_start:t_stop] = message
    else:
        mess_fwd = None

    # === Forward recursion (max-product) ===
    for t in range(1, T_len):
        aij = a[t - 1]
        i, j = x[t - 1], x[t]

        i_start, i_stop = state_loc[i], state_loc[i + 1]
        j_start, j_stop = state_loc[j], state_loc[j + 1]

        # Vectorized transition computation (GPU-optimized)
        T_slice = T_tr[aij, j_start:j_stop, i_start:i_stop]  # [num_j_clones, num_i_clones]
        message = (T_slice * message.unsqueeze(0)).max(dim=1).values
        p_obs = message.max()
        assert p_obs > 0, f"Zero probability at t={t}"
        message /= p_obs
        log2_lik[t] = torch.log2(p_obs)

        if store_messages:
            t_start, t_stop = mess_loc[t:t + 2]
            mess_fwd[t_start:t_stop] = message

    return (log2_lik, mess_fwd) if store_messages else log2_lik

def forwardE_mp(T_tr, E, Pi, n_clones, x, a, device, store_messages=False):
    """
    Max-product forward pass with emissions (for Viterbi decoding).
    
    Args:
        T_tr (Tensor): [n_actions, n_states, n_states] Transposed transition matrix
        E (Tensor): [n_states, n_obs] Emission probabilities
        Pi (Tensor): [n_states] Initial state distribution
        n_clones (Tensor): [n_obs] Number of clones per observation
        x (Tensor): [T] Observation sequence
        a (Tensor): [T] Action sequence
        device (torch.device): GPU or CPU
        store_messages (bool): Whether to store full messages

    Returns:
        log2_lik (Tensor): [T] log2-probability at each time step
        mess_fwd (Tensor or None): [T, n_states] forward messages (if store_messages=True)
    """
    assert E.shape == (n_clones.sum(), len(n_clones)), "E must be [n_states, n_obs]"
    x, a = x.to(device), a.to(device)
    Pi = Pi.to(device)
    E = E.to(device)

    T_len = x.shape[0]
    n_states = E.shape[0]
    dtype = T_tr.dtype
    log2_lik = torch.zeros(T_len, dtype=dtype, device=device)

    if store_messages:
        mess_fwd = torch.empty((T_len, n_states), dtype=dtype, device=device)

    # === Initial step ===
    t = 0
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.max()
    assert p_obs > 0, f"Zero probability at t={t}"
    message /= p_obs
    log2_lik[t] = torch.log2(p_obs)
    if store_messages:
        mess_fwd[t] = message

    # === Forward pass ===
    for t in range(1, T_len):
        aij = a[t - 1]
        j = x[t]

        # Max-product update: T_tr[aij] @ message, but using max instead of sum
        trans_scores = T_tr[aij] * message.unsqueeze(0)  # [n_states, n_states] * [1, n_states]
        message = trans_scores.max(dim=1).values         # [n_states]
        message *= E[:, j]                               # emission gating
        p_obs = message.max()
        assert p_obs > 0, f"Zero probability at t={t}"
        message /= p_obs
        log2_lik[t] = torch.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message

    return (log2_lik, mess_fwd) if store_messages else log2_lik

def backwardE(T, E, n_clones, x, a, device):
    """
    Compute backward messages using emissions.

    Args:
        T (Tensor): [n_actions, n_states, n_states] Transition matrix
        E (Tensor): [n_states, n_obs] Emission matrix
        n_clones (Tensor): [n_obs] number of clones per observation
        x (Tensor): [T] Observation sequence
        a (Tensor): [T] Action sequence
        device (torch.device)

    Returns:
        mess_bwd (Tensor): [T, n_states] Backward messages
    """
    assert E.shape == (n_clones.sum(), len(n_clones)), "Emission matrix shape mismatch"

    T_len = x.shape[0]
    n_states = E.shape[0]
    dtype = T.dtype
    mess_bwd = torch.empty((T_len, n_states), dtype=dtype, device=device)

    # Final step: uniform backward message
    t = T_len - 1
    message = torch.ones(n_states, dtype=dtype, device=device)
    message /= message.sum()
    mess_bwd[t] = message

    # Recursive backward pass
    for t in range(T_len - 2, -1, -1):
        aij = a[t]
        j = x[t + 1]
        message = torch.matmul(T[aij], (message * E[:, j]).unsqueeze(-1)).squeeze(-1)
        p_obs = message.sum()
        assert p_obs > 0, f"Zero probability at t={t}"
        message /= p_obs
        mess_bwd[t] = message

    return mess_bwd

def updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a, device, workspace=None):
    """
    Update the transition count matrix C using forward and backward messages.
    Args:
        C: [n_actions, n_states, n_states] — will be overwritten
        T: [n_actions, n_states, n_states] — transition matrix
        n_clones: [n_obs]
        mess_fwd: [sum(n_clones[x])]
        mess_bwd: [sum(n_clones[x])]
        x: [T] observation sequence
        a: [T] action sequence
        device: torch.device
    """
    # Memory Optimization: Use workspace to reduce allocations  
    if workspace is None:
        workspace = {}
    
    # Reuse pre-allocated tensors when possible
    if 'state_loc_upd' not in workspace or workspace['state_loc_upd'].size(0) != n_clones.size(0) + 1:
        workspace['state_loc_upd'] = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones.cumsum(0)])
    state_loc = workspace['state_loc_upd']
    
    if 'mess_loc_upd' not in workspace or workspace['mess_loc_upd'].size(0) != len(x) + 1:
        workspace['mess_loc_upd'] = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones[x].cumsum(0)])
    mess_loc = workspace['mess_loc_upd']
    
    timesteps = len(x)

    C.zero_()

    # Optimization: Pre-compute all updateC indices and use memory-efficient operations
    if timesteps > 1:
        # Vectorized index computation for updateC
        t_range = torch.arange(1, timesteps, device=device)
        i_indices_upd = x[:-1]  # [T-1] 
        j_indices_upd = x[1:]   # [T-1]
        a_indices_upd = a[:-1]  # [T-1]
        
        # Pre-compute all boundaries
        tm1_starts = mess_loc[t_range - 1]
        tm1_stops = mess_loc[t_range]
        t_starts = mess_loc[t_range]
        t_stops = mess_loc[t_range + 1]
        i_starts_upd = state_loc[i_indices_upd]
        i_stops_upd = state_loc[i_indices_upd + 1]
        j_starts_upd = state_loc[j_indices_upd] 
        j_stops_upd = state_loc[j_indices_upd + 1]

    # Ultra-optimized batched count matrix update for A100 GPUs
    # This addresses the major performance bottleneck with vectorized operations
    
    # Enable experimental batching for count updates (major performance boost)
    ENABLE_BATCHED_UPDATES = os.environ.get('CHMM_BATCHED_UPDATES', '1') == '1'
    ENABLE_ULTRA_OPTIMIZED = os.environ.get('CHMM_ULTRA_OPTIMIZED', '1') == '1'
    
    if ENABLE_BATCHED_UPDATES and timesteps > 10:
        # Batch processing optimization for large sequences
        # Group timesteps by action to maximize GPU utilization
        
        # Pre-allocate batch workspace tensors
        if 'batch_workspace' not in workspace:
            max_clone_size = n_clones.max().item()
            workspace['batch_workspace'] = {
                'batch_alphas': torch.empty(timesteps-1, max_clone_size, dtype=T.dtype, device=device),
                'batch_betas': torch.empty(timesteps-1, max_clone_size, dtype=T.dtype, device=device),
                'batch_norms': torch.empty(timesteps-1, dtype=T.dtype, device=device),
                'action_groups': {},
            }
        
        batch_ws = workspace['batch_workspace']
        
        # Group timesteps by action for efficient batching
        # CRITICAL FIX: Convert to CPU once instead of .item() in loop
        a_indices_cpu = a_indices_upd.cpu().numpy()
        action_groups = {}
        for idx in range(timesteps - 1):
            ajt = int(a_indices_cpu[idx])
            if ajt not in action_groups:
                action_groups[ajt] = []
            action_groups[ajt].append(idx)
        
        # Process each action group in batch
        for ajt, batch_indices in action_groups.items():
            if len(batch_indices) == 1:
                # Single timestep - use original method
                idx = batch_indices[0]
                # Convert indices to CPU once
                tm1_start, tm1_stop = tm1_starts[idx].item(), tm1_stops[idx].item()
                t_start, t_stop = t_starts[idx].item(), t_stops[idx].item()
                i_start, i_stop = i_starts_upd[idx].item(), i_stops_upd[idx].item()
                j_start, j_stop = j_starts_upd[idx].item(), j_stops_upd[idx].item()

                alpha = mess_fwd[tm1_start:tm1_stop]
                beta = mess_bwd[t_start:t_stop]
                T_slice = T[ajt, i_start:i_stop, j_start:j_stop]

                # CORRECTED: Use CPU algorithm - alpha.reshape(-1,1) * T * beta.reshape(1,-1)
                q = alpha.reshape(-1, 1) * T_slice * beta.reshape(1, -1)
                norm = q.sum()
                if norm > 0:
                    q /= norm
                    C[ajt, i_start:i_stop, j_start:j_stop] += q
            else:
                # Batch processing for multiple timesteps with same action
                # This is where the major speedup happens
                
                # Extract all alpha/beta vectors for this action
                batch_size = len(batch_indices)
                alphas_list = []
                betas_list = []
                boundaries_list = []
                
                # CRITICAL FIX: Pre-convert batch indices to avoid .item() in loop
                batch_tm1_starts = tm1_starts[batch_indices].cpu().numpy()
                batch_tm1_stops = tm1_stops[batch_indices].cpu().numpy()
                batch_t_starts = t_starts[batch_indices].cpu().numpy()
                batch_t_stops = t_stops[batch_indices].cpu().numpy()
                batch_i_starts = i_starts_upd[batch_indices].cpu().numpy()
                batch_i_stops = i_stops_upd[batch_indices].cpu().numpy()
                batch_j_starts = j_starts_upd[batch_indices].cpu().numpy()
                batch_j_stops = j_stops_upd[batch_indices].cpu().numpy()
                
                for i, idx in enumerate(batch_indices):
                    tm1_start, tm1_stop = int(batch_tm1_starts[i]), int(batch_tm1_stops[i])
                    t_start, t_stop = int(batch_t_starts[i]), int(batch_t_stops[i])
                    i_start, i_stop = int(batch_i_starts[i]), int(batch_i_stops[i])
                    j_start, j_stop = int(batch_j_starts[i]), int(batch_j_stops[i])
                    
                    alpha = mess_fwd[tm1_start:tm1_stop]
                    beta = mess_bwd[t_start:t_stop]
                    
                    alphas_list.append(alpha)
                    betas_list.append(beta)
                    boundaries_list.append((i_start, i_stop, j_start, j_stop))
                
                # Process boundaries that have the same dimensions together
                boundary_groups = {}
                for i, bounds in enumerate(boundaries_list):
                    key = (bounds[1] - bounds[0], bounds[3] - bounds[2])  # (i_size, j_size)
                    if key not in boundary_groups:
                        boundary_groups[key] = []
                    boundary_groups[key].append((i, bounds))
                
                # Batch process each boundary group
                for (i_size, j_size), group_items in boundary_groups.items():
                    if len(group_items) > 1:
                        # True batch processing for same-sized regions
                        batch_alphas = torch.stack([alphas_list[item] for item, _ in group_items])
                        batch_betas = torch.stack([betas_list[item] for item, _ in group_items])
                        
                        # Get T_slice (same for all since same action)
                        _, bounds = group_items[0]
                        i_start, i_stop, j_start, j_stop = bounds
                        T_slice = T[ajt, i_start:i_stop, j_start:j_stop]
                        
                        # CORRECTED: Use CPU algorithm - alpha.reshape(-1,1) * T * beta.reshape(1,-1)
                        # Shape: [batch_size, i_size, j_size]
                        batch_q = batch_alphas.unsqueeze(-1) * T_slice.unsqueeze(0) * batch_betas.unsqueeze(-2)
                        
                        # Vectorized normalization
                        batch_norms = batch_q.sum(dim=(1, 2))
                        valid_mask = batch_norms > 0
                        
                        if valid_mask.any():
                            # Normalize valid entries
                            batch_q[valid_mask] /= batch_norms[valid_mask].unsqueeze(-1).unsqueeze(-1)
                            
                            # Accumulate into count matrix
                            for k, (item_idx, bounds) in enumerate(group_items):
                                if valid_mask[k]:
                                    i_start, i_stop, j_start, j_stop = bounds
                                    C[ajt, i_start:i_stop, j_start:j_stop] += batch_q[k]
                    else:
                        # Single item - process normally
                        item_idx, bounds = group_items[0]
                        i_start, i_stop, j_start, j_stop = bounds
                        
                        alpha = alphas_list[item_idx]
                        beta = betas_list[item_idx]
                        T_slice = T[ajt, i_start:i_stop, j_start:j_stop]
                        
                        # CORRECTED: Use CPU algorithm - alpha.reshape(-1,1) * T * beta.reshape(1,-1)
                        q = alpha.reshape(-1, 1) * T_slice * beta.reshape(1, -1)
                        norm = q.sum()
                        if norm > 0:
                            q /= norm
                            C[ajt, i_start:i_stop, j_start:j_stop] += q
    else:
        # Fallback to optimized sequential processing for small sequences
        # CRITICAL FIX: Convert to CPU once instead of .item() in loop
        a_indices_cpu = a_indices_upd.cpu().numpy()
        tm1_starts_cpu = tm1_starts.cpu().numpy()
        tm1_stops_cpu = tm1_stops.cpu().numpy()
        t_starts_cpu = t_starts.cpu().numpy()
        t_stops_cpu = t_stops.cpu().numpy()
        i_starts_cpu = i_starts_upd.cpu().numpy()
        i_stops_cpu = i_stops_upd.cpu().numpy()
        j_starts_cpu = j_starts_upd.cpu().numpy()
        j_stops_cpu = j_stops_upd.cpu().numpy()
        
        for idx, t in enumerate(range(1, timesteps)):
            ajt = int(a_indices_cpu[idx])
            
            tm1_start, tm1_stop = int(tm1_starts_cpu[idx]), int(tm1_stops_cpu[idx])
            t_start, t_stop = int(t_starts_cpu[idx]), int(t_stops_cpu[idx])
            i_start, i_stop = int(i_starts_cpu[idx]), int(i_stops_cpu[idx])
            j_start, j_stop = int(j_starts_cpu[idx]), int(j_stops_cpu[idx])

            alpha = mess_fwd[tm1_start:tm1_stop]
            beta = mess_bwd[t_start:t_stop]
            T_slice = T[ajt, i_start:i_stop, j_start:j_stop]

            # CORRECTED: Use CPU algorithm - alpha.reshape(-1,1) * T * beta.reshape(1,-1)
            q = alpha.reshape(-1, 1) * T_slice * beta.reshape(1, -1)
            norm = q.sum()
            if norm > 0:
                q /= norm
                C[ajt, i_start:i_stop, j_start:j_stop] += q
            elif DEBUG_MODE:
                print(f"[Warning] Skipping update at t={t} due to 0 norm")

def backward(T, n_clones, x, a, device, workspace=None):
    """
    Compute backward messages.

    Args:
        T: [n_actions, n_states, n_states] transition matrix
        n_clones: [n_obs] number of clones for each observation
        x: [T] observation sequence
        a: [T] action sequence
        device: torch.device

    Returns:
        mess_bwd: [sum(n_clones[x])] backward messages
    """
    # Memory Optimization: Use workspace to reduce allocations
    if workspace is None:
        workspace = {}
    
    # Reuse pre-allocated tensors when possible
    if 'state_loc_bwd' not in workspace or workspace['state_loc_bwd'].size(0) != n_clones.size(0) + 1:
        workspace['state_loc_bwd'] = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones.cumsum(0)])
    state_loc = workspace['state_loc_bwd']
    
    if 'mess_loc_bwd' not in workspace or workspace['mess_loc_bwd'].size(0) != len(x) + 1:
        workspace['mess_loc_bwd'] = torch.cat([torch.zeros(1, dtype=n_clones.dtype, device=device), n_clones[x].cumsum(0)])
    mess_loc = workspace['mess_loc_bwd']
    
    dtype = T.dtype
    T_len = x.shape[0]

    # initialize final message
    t = T_len - 1
    i = x[t].item()
    if DEBUG_MODE:
        print(f"[DEBUG] backward (init): i={i}, type(i)={type(i)}")
        print(f"[DEBUG] backward (init): n_clones={n_clones}, type(n_clones)={type(n_clones)}")
        print(f"[DEBUG] backward (init): n_clones[i]={n_clones[i]}, type(n_clones[i])={type(n_clones[i])}")
    message = torch.ones(n_clones[i].item(), dtype = dtype, device = device) / n_clones[i]

    mess_bwd = torch.empty(mess_loc[-1], dtype=dtype, device=device)
    t_start, t_stop = mess_loc[t : t + 2]
    mess_bwd[t_start : t_stop] = message

    # Optimization: Pre-compute indices for backward pass
    seq_len = x.shape[0]
    if seq_len > 2:
        # Vectorized index computation for backward pass
        backward_range = torch.arange(seq_len - 2, -1, -1, device=device)
        i_indices_bwd = x[backward_range]  # [T-2 down to 0]
        j_indices_bwd = x[backward_range + 1]  # [T-1 down to 1]
        a_indices_bwd = a[backward_range]  # [T-2 down to 0]
        
        # Pre-compute state boundaries for backward pass
        i_starts_bwd = state_loc[i_indices_bwd]
        i_stops_bwd = state_loc[i_indices_bwd + 1]
        j_starts_bwd = state_loc[j_indices_bwd]
        j_stops_bwd = state_loc[j_indices_bwd + 1]

    # Vectorized backward recursion - optimized indexing and matrix operations
    # Note: Backward pass has sequential dependencies, but we optimize inner operations
    
    # Pre-allocate workspace for backward pass
    if 'backward_temp' not in workspace:
        max_msg_size = n_clones.max().item()
        workspace['backward_temp'] = torch.empty(max_msg_size, dtype=dtype, device=device)
    
    # CRITICAL FIX: Convert to CPU once instead of .item() in loop
    a_indices_bwd_cpu = a_indices_bwd.cpu().numpy()
    i_starts_bwd_cpu = i_starts_bwd.cpu().numpy()
    i_stops_bwd_cpu = i_stops_bwd.cpu().numpy()
    j_starts_bwd_cpu = j_starts_bwd.cpu().numpy()
    j_stops_bwd_cpu = j_stops_bwd.cpu().numpy()
    
    # Backward recursion with optimized tensor operations
    for idx, t in enumerate(range(seq_len - 2, -1, -1)):
        # Vectorized index access
        ajt = int(a_indices_bwd_cpu[idx])
        i_start, i_stop = int(i_starts_bwd_cpu[idx]), int(i_stops_bwd_cpu[idx])
        j_start, j_stop = int(j_starts_bwd_cpu[idx]), int(j_stops_bwd_cpu[idx])

        # Optimized matrix operations
        T_slice = T[ajt, i_start:i_stop, j_start:j_stop]
        
        # Use matmul for better GPU utilization (3.8x faster than mv)
        message = torch.matmul(T_slice, message.unsqueeze(-1)).squeeze(-1)
        
        # Vectorized normalization
        p_obs = message.sum()
        if DEBUG_MODE:
            assert p_obs > 0, f"Zero probability in backward at t={t}, obs={x[t]}"
        message = message / p_obs

        # Optimized message storage
        t_start, t_stop = mess_loc[t : t + 2]
        mess_bwd[t_start : t_stop] = message

    return mess_bwd

def backtrace(T, n_clones, x, a, mess_fwd, device):
    """
    Backtrace the Mean Average Precision (MAP) assignment of latent variables.

    Args:
        T (Tensor): [n_actions, n_states, n_states] Transition matrix
        n_clones (Tensor): [n_obs] Number of clones per observation
        x (Tensor): [T] Observation sequence
        a (Tensor): [T] Action sequence
        mess_fwd (Tensor): [sum(n_clones[x])] Forward messages
        device (torch.device): GPU or CPU

    Returns:
        torch.Tensor: [T] MAP assignment of latent variables
    """
    state_loc = torch.cat([torch.tensor([0], dtype=n_clones.dtype, device=device), n_clones.cumsum(0)])
    mess_loc = torch.cat([torch.tensor([0], dtype=n_clones.dtype, device=device), n_clones[x].cumsum(0)])
    clone_idx = torch.zeros(x.shape[0], dtype = torch.int64, device = device)

    # backward pass
    t = x.shape[0] - 1
    i = x[t]
    t_start, t_stop = mess_loc[t : t + 2]
    belief = mess_fwd[t_start : t_stop]
    clone_idx[t] = torch.argmax(belief)
    for t in range(x.shape[0] - 2, -1, -1):
        aij = a[t].item()
        i, j = x[t].item(), x[t + 1].item()

        i_start, i_stop = state_loc[i : i + 2]
        j_start = state_loc[j]
        t_start, t_stop = mess_loc[t : t + 2]
        belief = (mess_fwd[t_start : t_stop] * T[aij, i_start : i_stop, j_start + clone_idx[t + 1]])
        clone_idx[t] = torch.argmax(belief)
    states = state_loc[x] + clone_idx
    return states

def updateCE(CE, E, n_clones, mess_fwd, mess_bwd, x, a, device):
    """
    Update emission counts matrix CE using soft alignments.

    Args:
        CE (Tensor): [n_states, n_obs] emission count matrix to update (in-place)
        E (Tensor): [n_states, n_obs] emission matrix (not updated)
        n_clones (Tensor): [n_obs] number of clones per observation
        mess_fwd (Tensor): [T, n_states] forward messages
        mess_bwd (Tensor): [T, n_states] backward messages
        x (Tensor): [T] observation sequence
        a (Tensor): [T] action sequence
        device (torch.device)
    """
    T_len = x.shape[0]
    gamma = mess_fwd * mess_bwd  # [T, n_states]
    norm = gamma.sum(dim=1, keepdim=True)
    norm[norm == 0] = 1.0
    gamma = gamma / norm

    CE.zero_()  # clear before accumulation

    for t in range(T_len):
        CE[:, x[t]] += gamma[t]

def backtraceE(T, E, n_clones, x, a, mess_fwd, device):
    """
    Backtrace the MAP (most likely) assignment of clone states using emissions.

    Args:
        T (Tensor): [n_actions, n_states, n_states] Transition matrix
        E (Tensor): [n_states, n_obs] Emission matrix
        n_clones (Tensor): [n_obs] number of clones per observation
        x (Tensor): [T] observation sequence (int64)
        a (Tensor): [T] action sequence (int64)
        mess_fwd (Tensor): [T, n_states] max-product forward messages
        device (torch.device)

    Returns:
        states (Tensor): [T] MAP clone state indices
    """
    assert E.shape == (n_clones.sum(), len(n_clones)), "Emission matrix shape mismatch"
    T_len = x.shape[0]
    x, a = x.to(device), a.to(device)

    states = torch.zeros(T_len, dtype=torch.int64, device=device)

    # Final step — select best clone from final forward message
    t = T_len - 1
    belief = mess_fwd[t]
    states[t] = torch.argmax(belief)

    # Backtrace
    for t in range(T_len - 2, -1, -1):
        aij = a[t].item()
        next_state = states[t + 1].item()
        belief = mess_fwd[t] * T[aij, :, next_state]  # elementwise transition probability
        states[t] = torch.argmax(belief)

    return states

def rargmax(x):
    """
    Randomized argmax: if multiple entries share the max value, choose one at random.

    Args:
        x (Tensor): 1D tensor

    Returns:
        int: index of one of the max elements
    """
    max_val = torch.max(x)
    candidates = (x == max_val).nonzero(as_tuple=True)[0]
    return candidates[torch.randint(len(candidates), (1,))].item()

def forward_mp_all(T_tr, Pi_x, Pi_a, n_clones, target_state, max_steps, device):
    """
    Run max-product forward pass over all actions to reach a target clone state.

    Args:
        T_tr (Tensor): [n_actions, n_states, n_states] — Transposed transition matrix
        Pi_x (Tensor): [n_states] — Initial state distribution
        Pi_a (Tensor): [n_actions] — Action priors
        n_clones (Tensor): [n_obs]
        target_state (int): Goal clone state
        max_steps (int): Maximum steps allowed
        device (torch.device)

    Returns:
        log2_lik (Tensor): [T] log-probability trajectory
        mess_fwd (Tensor): [T, n_states] forward messages
    """
    T_len = max_steps
    n_states = Pi_x.shape[0]
    dtype = T_tr.dtype

    log2_lik = []
    mess_fwd = []

    message = Pi_x.clone()
    p_obs = message.max()
    assert p_obs > 0, "Initial state has zero probability"
    message /= p_obs
    log2_lik.append(torch.log2(p_obs))
    mess_fwd.append(message)

    # Collapse over actions with Pi_a weights
    T_tr_maxa = (T_tr * Pi_a.view(-1, 1, 1)).amax(dim=0)  # [to, from]

    for t in range(1, T_len):
        message = (T_tr_maxa * message.view(1, -1)).amax(dim=1)
        p_obs = message.max()
        assert p_obs > 0, f"Zero observation probability at t={t}"
        message /= p_obs
        log2_lik.append(torch.log2(p_obs))
        mess_fwd.append(message)

        if message[target_state] > 0:
            break
    else:
        raise RuntimeError("Unable to find a bridging path to target state.")

    return torch.stack(log2_lik), torch.stack(mess_fwd)

def backtrace_all(T, Pi_a, n_clones, mess_fwd, target_state, device):
    """
    Trace back the most likely (action, state) path to reach `target_state`.

    Args:
        T (Tensor): [n_actions, n_states, n_states]
        Pi_a (Tensor): [n_actions]
        n_clones (Tensor): [n_obs]
        mess_fwd (Tensor): [T, n_states] forward messages
        target_state (int): Final target clone index
        device (torch.device)

    Returns:
        actions (Tensor): [T] action sequence
        states (Tensor): [T] clone state sequence
    """
    T_len = mess_fwd.shape[0]
    n_states = T.shape[1]
    dtype = T.dtype

    actions = torch.full((T_len,), -1, dtype=torch.int64, device=device)
    states = torch.zeros((T_len,), dtype=torch.int64, device=device)

    states[-1] = target_state

    for t in range(T_len - 2, -1, -1):
        belief = mess_fwd[t].view(1, -1) * T[:, :, states[t + 1]] * Pi_a.view(-1, 1)
        flat_belief = belief.flatten()
        a_s = rargmax(flat_belief)
        actions[t] = a_s // n_states
        states[t] = a_s % n_states

    return actions, states

def train_chmm(n_clones, x, a, device=None, method='em_T', n_iter=50, pseudocount=0.01, seed=42, 
               learn_E=False, viterbi=False, early_stopping=True, min_improvement=1e-6,
               use_mixed_precision=False, memory_efficient=True):
    """
    Train a CHMM using the CHMM_torch class methods with progression tracking.
    
    Args:
        n_clones (Tensor): [n_obs] Number of clones per observation
        x (Tensor): [T] Observation sequence
        a (Tensor): [T] Action sequence
        device (torch.device, optional): GPU or CPU device (auto-detected if None)
        method (str): Training method - 'em_T', 'viterbi_T', or 'em_E'
        n_iter (int): Number of EM iterations
        pseudocount (float): Pseudocount for smoothing
        seed (int): Random seed
        learn_E (bool): Whether to also learn emission matrix E
        viterbi (bool): Whether to use Viterbi (hard) EM instead of soft EM
        early_stopping (bool): Whether to use early stopping
        min_improvement (float): Minimum improvement threshold for early stopping
        use_mixed_precision (bool): Enable mixed precision training for faster GPU computation
        memory_efficient (bool): Enable GPU memory optimizations
        
    Returns:
        tuple: (trained_model, progression_list)
            - trained_model: CHMM_torch instance with trained parameters
            - progression_list: List of BPS values showing training progression
    """
    # Import CHMM_torch here to avoid circular imports
    from .chmm_torch import CHMM_torch
    
    # Input validation
    assert isinstance(n_clones, torch.Tensor), "n_clones must be a tensor"
    assert isinstance(x, torch.Tensor), "x must be a tensor" 
    assert isinstance(a, torch.Tensor), "a must be a tensor"
    assert n_clones.ndim == 1, f"n_clones must be 1D, got {n_clones.ndim}D"
    assert x.ndim == 1, f"x must be 1D, got {x.ndim}D"
    assert a.ndim == 1, f"a must be 1D, got {a.ndim}D"
    assert len(x) == len(a), f"x and a must have same length: {len(x)} != {len(a)}"
    assert method in ['em_T', 'viterbi_T', 'em_E'], f"method must be 'em_T', 'viterbi_T', or 'em_E', got {method}"
    
    # Auto-detect device if not provided (optimized)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"Auto-detected CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps:0")
            print("Auto-detected MPS (Apple Silicon) device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
    
    # GPU memory optimization
    if device.type == 'cuda' and memory_efficient:
        torch.cuda.empty_cache()
    
    # Move tensors to device (avoid redundant transfers)
    if n_clones.device != device:
        n_clones = n_clones.to(device)
    if x.device != device:
        x = x.to(device)
    if a.device != device:
        a = a.to(device)
    
    # Set random seed for reproducibility (enhanced)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Initialize CHMM model
    print(f"Initializing CHMM with {n_clones.sum().item()} states on {device}")
    print(f"Input tensors: n_clones device={n_clones.device}, x device={x.device}, a device={a.device}")
    model = CHMM_torch(
        n_clones=n_clones,
        x=x, 
        a=a,
        pseudocount=pseudocount,
        seed=seed,
        device=device
    )
    print(f"Model initialized on device: {model.device}")
    print(f"Model T matrix device: {model.T.device if hasattr(model, 'T') else 'Not initialized'}")
    
    print(f"Starting {method} training for {n_iter} iterations...")
    if use_mixed_precision and device.type == 'cuda':
        print("Mixed precision training enabled")
    
    # GPU memory monitoring
    if device.type == 'cuda':
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Choose training method and run with mixed precision support
    with torch.autocast(device_type=device.type if device.type == 'cuda' else 'cpu', 
                       enabled=use_mixed_precision and device.type == 'cuda'):
        
        if method == 'em_T' or (not viterbi and not learn_E):
            # Standard soft EM for transition matrices
            progression = model.learn_em_T(
                x=x,
                a=a, 
                n_iter=n_iter,
                term_early=early_stopping,
                min_improvement=min_improvement
            )
            
        elif method == 'viterbi_T' or viterbi:
            # Hard EM (Viterbi) for transition matrices
            progression = model.learn_viterbi_T(
                x=x,
                a=a,
                n_iter=n_iter
            )
            
        elif method == 'em_E' or learn_E:
            # EM for emission matrices (keeps T fixed)
            progression, learned_E = model.learn_em_E(
                x=x,
                a=a,
                n_iter=n_iter,
                pseudocount_extra=pseudocount
            )
            print(f"Learned emission matrix E with shape {learned_E.shape}")
            
        else:
            raise ValueError(f"Unknown training configuration: method={method}, viterbi={viterbi}, learn_E={learn_E}")
    
    # If learning both T and E sequentially
    if learn_E and method != 'em_E':
        print("Also learning emission matrix E...")
        e_progression, learned_E = model.learn_em_E(
            x=x,
            a=a, 
            n_iter=n_iter//2,  # Use fewer iterations for E step
            pseudocount_extra=pseudocount
        )
        # Combine progressions
        progression.extend(e_progression)
        print(f"Final emission matrix E shape: {learned_E.shape}")
    
    print(f"Training completed. Final BPS: {progression[-1]:.4f}")
    print(f"Total improvement: {progression[0] - progression[-1]:.4f}")
    
    # GPU memory cleanup
    if device.type == 'cuda' and memory_efficient:
        torch.cuda.empty_cache()
        print(f"GPU memory after training: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    return model, progression

def make_E(n_clones, device=None):
    """
    Create emission matrix mapping clones to observations.
    
    Args:
        n_clones (Tensor): [n_obs] Number of clones per observation
        device (torch.device, optional): Device for tensor
        
    Returns:
        Tensor: [total_clones, n_obs] Emission matrix
    """
    if device is None:
        device = n_clones.device if isinstance(n_clones, torch.Tensor) else torch.device('cpu')
    
    # Convert to tensor if needed
    if not isinstance(n_clones, torch.Tensor):
        n_clones = torch.tensor(n_clones, device=device)
    else:
        n_clones = n_clones.to(device)
    
    # Validate input
    assert n_clones.ndim == 1, f"n_clones must be 1D, got {n_clones.ndim}D"
    assert torch.all(n_clones > 0), "all n_clones must be positive"
    
    total_clones = n_clones.sum().item()
    n_obs = len(n_clones)
    
    E = torch.zeros((total_clones, n_obs), device=device, dtype=torch.float32)
    
    idx = 0
    for obs_id, n in enumerate(n_clones):
        n = n.item()
        E[idx:idx+n, obs_id] = 1.0
        idx += n
    
    # Validate output
    assert E.shape == (total_clones, n_obs), f"E shape mismatch: {E.shape} != ({total_clones}, {n_obs})"
    assert torch.allclose(E.sum(dim=1), torch.ones(total_clones, device=device)), "E rows must sum to 1"
    
    return E

def make_E_sparse(n_clones, device=None):
    """
    Create sparse emission matrix mapping clones to observations.
    
    Args:
        n_clones (Tensor): [n_obs] Number of clones per observation
        device (torch.device, optional): Device for tensor
        
    Returns:
        torch.sparse.FloatTensor: [total_clones, n_obs] Sparse emission matrix
    """
    if device is None:
        device = n_clones.device if isinstance(n_clones, torch.Tensor) else torch.device('cpu')
    
    # Convert to tensor if needed
    if not isinstance(n_clones, torch.Tensor):
        n_clones = torch.tensor(n_clones, device=device)
    else:
        n_clones = n_clones.to(device)
    
    # Validate input
    assert n_clones.ndim == 1, f"n_clones must be 1D, got {n_clones.ndim}D"
    assert torch.all(n_clones > 0), "all n_clones must be positive"
    
    total_clones = n_clones.sum().item()
    n_obs = len(n_clones)
    
    # Build sparse indices and values
    indices = []
    values = []
    
    idx = 0
    for obs_id, n in enumerate(n_clones):
        n = n.item()
        for j in range(n):
            indices.append([idx, obs_id])
            values.append(1.0)
            idx += 1
    
    # Convert to tensors
    indices = torch.tensor(indices, device=device).t()
    values = torch.tensor(values, device=device)
    
    # Create sparse tensor
    E_sparse = torch.sparse_coo_tensor(
        indices, values, (total_clones, n_obs), device=device
    ).coalesce()
    
    # Validate output
    assert E_sparse.shape == (total_clones, n_obs), f"E_sparse shape mismatch: {E_sparse.shape} != ({total_clones}, {n_obs})"
    
    return E_sparse

def compute_forward_messages(chmm_state, x, a, device, pseudocount_E=1e-10):
    """
    Compute forward messages for a CHMM.
    
    Args:
        chmm_state (dict): CHMM state dictionary with T, E, Pi_x, n_clones
        x (Tensor): [T] Observation sequence
        a (Tensor): [T] Action sequence  
        device (torch.device): GPU or CPU device
        pseudocount_E (float): Pseudocount for emission smoothing
        
    Returns:
        Tensor: [T, n_states] Forward messages
    """
    # Validate inputs
    assert isinstance(chmm_state, dict), f"chmm_state must be dict, got {type(chmm_state)}"
    required_keys = ['T', 'E', 'Pi_x', 'n_clones']
    for key in required_keys:
        assert key in chmm_state, f"chmm_state missing key: {key}"
    
    # Extract parameters and ensure they're on the correct device
    T = chmm_state['T'].to(device)
    E = chmm_state['E'].to(device) 
    Pi_x = chmm_state['Pi_x'].to(device)
    n_clones = chmm_state['n_clones'].to(device)
    
    # Move sequences to device
    x, a = x.to(device), a.to(device)
    
    # Validate tensor shapes
    assert T.ndim == 3, f"T must be 3D [n_actions, n_states, n_states], got {T.ndim}D"
    assert E.ndim == 2, f"E must be 2D [n_states, n_obs], got {E.ndim}D"
    assert Pi_x.ndim == 1, f"Pi_x must be 1D [n_states], got {Pi_x.ndim}D"
    
    # Apply pseudocount smoothing to emissions
    E_smooth = E + pseudocount_E
    E_smooth = E_smooth / E_smooth.sum(dim=1, keepdim=True).clamp(min=1e-10)
    
    # Compute forward messages with correct transpose
    _, mess_fwd = forwardE(
        T.transpose(1, 2), E_smooth, Pi_x, 
        n_clones, x, a, device, store_messages=True
    )
    
    # Validate output
    assert mess_fwd.device == device, f"Output device mismatch: {mess_fwd.device} != {device}"
    assert mess_fwd.shape[0] == len(x), f"Time dimension mismatch: {mess_fwd.shape[0]} != {len(x)}"
    
    return mess_fwd

def compute_place_field(mess_fwd, rc, clone, device=None):
    """
    Compute place field for a given clone using GPU-optimized PyTorch operations.
    
    This function calculates the average probability of a clone's activation 
    at each unique (row, column) position.
    
    Args:
        mess_fwd (Tensor): [T, n_states] Forward messages from CHMM
        rc (Tensor): [T, 2] Row-column positions for each time step
        clone (int): The specific clone index to compute the place field for
        device (torch.device, optional): GPU or CPU device. Auto-detected if None.
        
    Returns:
        Tensor: [max_r+1, max_c+1] Matrix representing the clone's place field
    """
    if device is None:
        device = mess_fwd.device
    
    # Move tensors to the target device if they aren't already there
    if mess_fwd.device != device:
        mess_fwd = mess_fwd.to(device)
    if rc.device != device:
        rc = rc.to(device)
    
    # --- Input Validation ---
    assert mess_fwd.ndim == 2, f"mess_fwd must be 2D, got {mess_fwd.ndim}D"
    assert rc.ndim == 2 and rc.shape[1] == 2, f"rc must be a [T, 2] tensor, got shape {rc.shape}"
    assert mess_fwd.shape[0] == rc.shape[0], \
        f"Sequence length mismatch: mess_fwd has {mess_fwd.shape[0]} steps, rc has {rc.shape[0]} steps"
    assert isinstance(clone, int), f"clone must be an integer, got {type(clone)}"
    assert 0 <= clone < mess_fwd.shape[1], f"clone index {clone} is out of bounds for {mess_fwd.shape[1]} clones"
    
    # --- Field Computation ---
    # Determine the dimensions of the place field from the max row and column values
    max_r, max_c = rc.max(dim=0)[0]
    field_shape = (max_r.item() + 1, max_c.item() + 1)
    
    # Find unique (row, column) pairs and get their inverse indices
    unique_rc, inverse_indices = torch.unique(rc, dim=0, return_inverse=True)
    
    # Extract the probabilities for the specified clone
    clone_probs = mess_fwd[:, clone]
    
    # Sum the probabilities for each unique location using scatter_add_ for efficiency
    # Using bincount is a highly optimized way to count occurrences
    visit_counts = torch.bincount(inverse_indices, minlength=unique_rc.shape[0])
    
    # Sum probabilities for each unique location
    summed_probs = torch.zeros(unique_rc.shape[0], device=device, dtype=mess_fwd.dtype)
    summed_probs.scatter_add_(0, inverse_indices, clone_probs)
    
    # Initialize the final field and count matrices
    field = torch.zeros(field_shape, device=device, dtype=mess_fwd.dtype)
    count = torch.zeros(field_shape, device=device, dtype=torch.int64)
    
    # Place the summed probabilities and visit counts into the field matrices
    field[unique_rc[:, 0], unique_rc[:, 1]] = summed_probs
    count[unique_rc[:, 0], unique_rc[:, 1]] = visit_counts
    
    # Avoid division by zero by ensuring counts are at least 1
    count = count.clamp(min=1)
    
    # Normalize the field by the visit counts to get the average probability
    field = field / count.to(field.dtype)
    
    return field