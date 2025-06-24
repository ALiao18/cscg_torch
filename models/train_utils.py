from __future__ import print_function
from builtins import range
import numpy as np
from tqdm import trange
import sys
import torch

def validate_seq(x, a, n_clones = None):
    """
    validate the sequence of observations and actions
    """
    assert len(x) == len(a) > 0
    assert len(x.shape) == len(a.shape) == 1, "Flatten your array first"
    assert x.dtype == a.dtype == torch.int64, "both observations and actions must discrete values"
    assert 0 <= x.min(), "Number of emissions inconsistent with training sequence"
    if n_clones is not None:
        assert len(n_clones.shape) == 1, "Flatten your array first"
        assert n_clones.dtype == torch.int64, "n_clones must be discrete int"
        assert torch.all(n_clones > 0), "You can't provide zero clones for any emission"
        n_emissions = n_clones.shape[0]
        assert x.max().item() < n_emissions, "Number of emissions inconsistent with training sequence"

def forward(T_tr, Pi, n_clones, x, a, device, store_messages = False):
    """
    Log-probability of a sequence, and optionally, messages

    T_tr: transition matrix transposed
    Pi: initial state distribution
    n_clones: number of clones for each emission
    x: observation sequence
    a: action sequence
    store_messages: whether to store messages
    """
    # Ensure all tensors are on the correct device
    x, a = x.to(device), a.to(device)
    Pi, n_clones = Pi.to(device), n_clones.to(device)
    T_tr = T_tr.to(device)
    
    # compute state and message locations
    state_loc = torch.cat([torch.tensor([0], dtype=n_clones.dtype, device=device), n_clones]).cumsum(0)
    T_len = T_tr.shape[1]
    log2_lik = torch.zeros(len(x), dtype = T_tr.dtype, device = device)
    
    # initialize messages
    t = 0
    j = x[t]
    j_start, j_stop = state_loc[j : j + 2]
    message = Pi[j_start:j_stop].clone()
    p_obs = message.sum()
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

    # forward pass
    for t in range(1, x.shape[0]):
        ajt = a[t - 1]
        i, j = x[t - 1], x[t]
        i_start, i_stop = state_loc[i : i + 2]
        j_start, j_stop = state_loc[j : j + 2]

        # matrix vector multiplication
        T_tr_slice = T_tr[ajt, j_start:j_stop, i_start:i_stop]
        message = torch.matmul(T_tr_slice, message)
        p_obs = message.sum()
        assert p_obs > 0, "Probability of observation is zero"
        message = message / p_obs
        log2_lik[t] = torch.log2(p_obs)

        if store_messages:
            t_start, t_stop = mess_loc[t : t + 2]
            mess_fwd[t_start:t_stop] = message
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

def updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a, device):
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
    state_loc = torch.cat([torch.tensor([0], dtype = n_clones.dtype, device = device), n_clones.cumsum(0)])
    mess_loc = torch.cat([torch.tensor([0], dtype = n_clones.dtype, device = device), n_clones[x].cumsum(0)])
    timesteps = len(x)

    C.zero_()

    # forward pass?? I'm, unsure about this
    for t in range(1, timesteps):
        ajt = a[t - 1]
        i, j= x[t - 1], x[t]

        tm1_start, tm1_stop = mess_loc[t - 1 : t + 1]
        t_start, t_stop = mess_loc[t : t + 2]
        i_start, i_stop = state_loc[i : i + 2]
        j_start, j_stop = state_loc[j : j + 2]

        alpha = mess_fwd[tm1_start:tm1_stop].reshape(-1, 1)        # [num_i_clones, 1]
        beta = mess_bwd[t_start:t_stop].reshape(1, -1)             # [1, num_j_clones]
        T_slice = T[ajt, i_start:i_stop, j_start:j_stop]           # [num_i_clones, num_j_clones]

        q = alpha * T_slice * beta                                 # [num_i_clones, num_j_clones]
        norm = q.sum()
        if norm > 0:
            q /= norm
            C[ajt, i_start:i_stop, j_start:j_stop] += q
        else:
            print(f"[Warning] Skipping update at t={t} due to 0 norm")

def backward(T, n_clones, x, a, device):
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
    state_loc = torch.cat([torch.tensor([0], dtype=n_clones.dtype, device=device), n_clones.cumsum(0)])
    mess_loc = torch.cat([torch.tensor([0], dtype=n_clones.dtype, device=device), n_clones[x].cumsum(0)])
    dtype = T.dtype
    T_len = x.shape[0]

    # initialize final message
    t = T_len - 1
    i = x[t]
    message = torch.ones(n_clones[i], dtype = dtype, device = device) / n_clones[i]

    mess_bwd = torch.empty(mess_loc[-1], dtype=dtype, device=device)
    t_start, t_stop = mess_loc[t : t + 2]
    mess_bwd[t_start : t_stop] = message

    # backward recursion
    for t in range(x.shape[0] - 2, -1, -1):
        ajt = a[t]
        i = x[t]
        j = x[t + 1]
        i_start, i_stop = state_loc[i : i + 2]
        j_start, j_stop = state_loc[j : j + 2]

        T_slice = T[ajt, i_start:i_stop, j_start:j_stop]
        message = torch.matmul(T_slice, message)
        p_obs = message.sum()
        assert p_obs > 0, f"Zero probability in backward at t={t}, obs={i}"
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

