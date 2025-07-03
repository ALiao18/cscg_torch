from __future__ import print_function
from builtins import range
import numpy as np
from tqdm import trange
import sys
import torch
import os

# Debug mode control - set CHMM_DEBUG=1 to enable assertions
DEBUG_MODE = os.environ.get('CHMM_DEBUG', '0') == '1'

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
    
    # Ensure all tensors are on the correct device (optimized)
    if x.device != device:
        x = x.to(device)
    if a.device != device:
        a = a.to(device)
    if Pi.device != device:
        Pi = Pi.to(device)
    if n_clones.device != device:
        n_clones = n_clones.to(device)
    if T_tr.device != device:
        T_tr = T_tr.to(device)
    
    # V100 Memory Optimization: Use workspace to reduce allocations
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
        mess_fwd = torch.empty(mess_loc[-1], dtype = T_tr.dtype, device = device)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd[t_start:t_stop] = message
    else:
        mess_fwd = None

    # V100 Optimization: Pre-compute all indices for vectorized access
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
    
    # forward pass - optimized for V100
    for t in range(1, seq_len):
        # Use pre-computed indices
        ajt = a_indices[t-1].item()
        i_start, i_stop = i_starts[t-1].item(), i_stops[t-1].item()
        j_start, j_stop = j_starts[t-1].item(), j_stops[t-1].item()

        # matrix vector multiplication (GPU optimized with index_select for better memory access)
        T_tr_slice = T_tr[ajt, j_start:j_stop, i_start:i_stop]
        message = torch.mv(T_tr_slice, message)  # Use mv instead of matmul for matrix-vector
        
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
        message = torch.matmul(T_tr[aij], message)
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
        message = T[aij].matmul(message * E[:, j])
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
    # V100 Memory Optimization: Use workspace to reduce allocations  
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

    # V100 Optimization: Pre-compute all updateC indices and use memory-efficient operations
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

    # Memory-optimized count matrix update
    for idx, t in enumerate(range(1, timesteps)):
        # Use pre-computed indices
        ajt = a_indices_upd[idx].item()
        
        tm1_start, tm1_stop = tm1_starts[idx].item(), tm1_stops[idx].item()
        t_start, t_stop = t_starts[idx].item(), t_stops[idx].item()
        i_start, i_stop = i_starts_upd[idx].item(), i_stops_upd[idx].item()
        j_start, j_stop = j_starts_upd[idx].item(), j_stops_upd[idx].item()

        # Memory-efficient tensor operations using outer product
        alpha = mess_fwd[tm1_start:tm1_stop]                       # [num_i_clones]
        beta = mess_bwd[t_start:t_stop]                            # [num_j_clones]
        T_slice = T[ajt, i_start:i_stop, j_start:j_stop]           # [num_i_clones, num_j_clones]

        # Use outer product for better V100 utilization: alpha[:, None] * beta[None, :] * T_slice
        q = torch.outer(alpha, beta) * T_slice                     # More efficient than multiple torch.mul calls
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
    # V100 Memory Optimization: Use workspace to reduce allocations
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
    i = x[t]
    message = torch.ones(n_clones[i], dtype = dtype, device = device) / n_clones[i]

    mess_bwd = torch.empty(mess_loc[-1], dtype=dtype, device=device)
    t_start, t_stop = mess_loc[t : t + 2]
    mess_bwd[t_start : t_stop] = message

    # V100 Optimization: Pre-compute indices for backward pass
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

    # backward recursion - optimized for V100
    for idx, t in enumerate(range(seq_len - 2, -1, -1)):
        # Use pre-computed indices
        ajt = a_indices_bwd[idx].item()
        i_start, i_stop = i_starts_bwd[idx].item(), i_stops_bwd[idx].item()
        j_start, j_stop = j_starts_bwd[idx].item(), j_stops_bwd[idx].item()

        T_slice = T[ajt, i_start:i_stop, j_start:j_stop]
        # Ensure message is on same device as T_slice for device compatibility
        if message.device != T_slice.device:
            message = message.to(T_slice.device)
        message = torch.mv(T_slice, message)  # Use mv for matrix-vector multiplication
        p_obs = message.sum()
        if DEBUG_MODE:
            assert p_obs > 0, f"Zero probability in backward at t={t}, obs={x[t]}"
        message = message / p_obs

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
        aij = a[t]
        i, j = x[t], x[t + 1]

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
        aij = a[t]
        next_state = states[t + 1]
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
        seed=seed
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

def place_field(mess_fwd, rc, clone, device=None):
    """
    Compute place field for a given clone using GPU-optimized operations.
    
    Args:
        mess_fwd (Tensor): [T, n_states] Forward messages
        rc (Tensor): [T, 2] Row-column positions
        clone (int): Clone index
        device (torch.device, optional): GPU or CPU device
        
    Returns:
        Tensor: [max_r+1, max_c+1] Place field matrix
    """
    if device is None:
        device = mess_fwd.device
    
    # Move to device efficiently
    if mess_fwd.device != device:
        mess_fwd = mess_fwd.to(device)
    if rc.device != device:
        rc = rc.to(device)
    
    # Validate inputs
    assert mess_fwd.ndim == 2, f"mess_fwd must be 2D, got {mess_fwd.ndim}D"
    assert rc.ndim == 2 and rc.shape[1] == 2, f"rc must be [T, 2], got {rc.shape}"
    assert mess_fwd.shape[0] == rc.shape[0], f"sequence length mismatch: {mess_fwd.shape[0]} != {rc.shape[0]}"
    assert isinstance(clone, int), f"clone must be int, got {type(clone)}"
    assert 0 <= clone < mess_fwd.shape[1], f"clone {clone} out of range [0, {mess_fwd.shape[1]})"
    
    # Get field dimensions
    max_r, max_c = rc.max(dim=0)[0]
    field_shape = (max_r.item() + 1, max_c.item() + 1)
    
    # Initialize field and count matrices
    field = torch.zeros(field_shape, device=device, dtype=mess_fwd.dtype)
    count = torch.zeros(field_shape, device=device, dtype=torch.int64)
    
    # GPU-optimized accumulation using scatter_add for better performance
    flat_indices = rc[:, 0] * field_shape[1] + rc[:, 1]  # Convert 2D indices to 1D
    
    # Flatten field matrices for scatter operations
    field_flat = field.flatten()
    count_flat = count.flatten()
    
    # Accumulate using scatter_add (GPU optimized)
    field_flat.scatter_add_(0, flat_indices, mess_fwd[:, clone])
    count_flat.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.int64))
    
    # Reshape back to 2D
    field = field_flat.view(field_shape)
    count = count_flat.view(field_shape)
    
    # Avoid division by zero
    count = count.clamp(min=1)
    
    # Normalize by visit counts
    field = field / count.float()
    
    return field

def train_chmm(n_clones, x, a, device=None, method='em_T', n_iter=50, pseudocount=0.01, seed=42, 
               learn_E=False, viterbi=False, early_stopping=True, min_improvement=1e-6,
               use_mixed_precision=False, memory_efficient=True):
    """
    Train a CHMM model with GPU optimizations.
    
    Args:
        n_clones: Number of clones per observation
        x: Observation sequence
        a: Action sequence  
        device: Device for computation
        method: Training method ('em_T', 'viterbi_T', 'em_E')
        n_iter: Number of iterations
        pseudocount: Pseudocount for smoothing
        seed: Random seed
        learn_E: Whether to learn emissions
        viterbi: Whether to use Viterbi training
        early_stopping: Whether to use early stopping
        min_improvement: Minimum improvement for early stopping
        use_mixed_precision: Whether to use mixed precision (deprecated for A100)
        memory_efficient: Whether to use memory optimizations
        
    Returns:
        tuple: (model, progression) where progression is BPS history
    """
    # Import here to avoid circular imports
    from .chmm_torch import CHMM_torch
    
    # Device setup
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    # Convert inputs to tensors on correct device
    if not isinstance(n_clones, torch.Tensor):
        n_clones = torch.tensor(n_clones, dtype=torch.int64, device=device)
    else:
        n_clones = n_clones.to(device)
        
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.int64, device=device)
    else:
        x = x.to(device)
        
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.int64, device=device)
    else:
        a = a.to(device)
    
    # Initialize CHMM model with optimizations
    model = CHMM_torch(
        n_clones=n_clones, 
        x=x, 
        a=a,
        pseudocount=pseudocount,
        seed=seed,
        memory_efficient=memory_efficient
    )
    
    # Run training with appropriate method
    if method == 'em_T':
        progression = model.learn_em_T(
            x=x, 
            a=a, 
            n_iter=n_iter, 
            term_early=early_stopping,
            min_improvement=min_improvement
        )
    elif method == 'viterbi_T':
        progression = model.learn_viterbi_T(
            x=x, 
            a=a, 
            n_iter=n_iter, 
            term_early=early_stopping
        )
    elif method == 'em_E':
        if learn_E:
            progression = model.learn_em_E(
                x=x, 
                a=a, 
                n_iter=n_iter, 
                term_early=early_stopping
            )
        else:
            raise ValueError("learn_E must be True when using method='em_E'")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return model, progression