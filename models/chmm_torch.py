from __future__ import print_function
from builtins import range
import numpy as np
from tqdm import trange
import sys
import torch

from .train_utils import (validate_seq, forward, forwardE, forward_mp, 
                          forwardE_mp,backward, updateC, backtrace, 
                          backtraceE, backwardE, updateCE, forward_mp_all, 
                          backtrace_all)

class CHMM_torch(object):
    def __init__(self, n_clones, x, a, pseudocount = 0.0, dtype = torch.float32, seed = 42):
        """
        Construct a CHMM object. 

        n_clones: array where n_clones[i] is the number of clones assigned to observation i
        x: observation sequence
        a: action sequence
        pseudocount: pseudocount for the transition matrix
        """
        np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==== Validate Sequence ====
        self.n_clones = n_clones
        validate_seq(x, a, self.n_clones)
        assert pseudocount >= 0.0, "The pseudocount should be non-negative"

        # ==== Initialize Parameters ====
        self.dtype = dtype
        self.pseudocount = pseudocount
        n_states = self.n_clones.sum()
        n_actions = a.max() + 1 
        self.C = torch.rand(n_actions, n_states, n_states, device = self.device).to(dtype)
        self.Pi_x = torch.ones(n_states, dtype = dtype, device = self.device) / n_states
        self.Pi_a = torch.ones(n_actions, dtype = dtype, device = self.device) / n_actions
        self.update_T()

        # ==== Print Summary ====
        print("Average number of clones:", n_clones.float().mean().item())
        print("C device:", self.C.device) # ensures tensors are on the correct device
        print("Pi_x device:", self.Pi_x.device)
        
    def update_T(self, verbose = True):
        """
        Update transition matrix given the accumulated counts matrix
        """
        self.T = self.C + self.pseudocount
        norm = self.T.sum(dim=2, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        self.T = self.T / norm

        if verbose:
            # ==== Print Summary ====
            print("T shape: ", self.T.shape)
            print("T device: ", self.T.device)
            print("T sum along dim=2:", self.T.sum(dim=2))

    def update_E(self, CE):
        """
        Update the emission matrix given the accumulated counts matrix

        CE: emissions count matrix
        """
        CE = CE.to(dtype = self.dtype, device = self.device)
        E = CE + self.pseudocount

        norm = E.sum(1, keepdims =  True)
        norm[norm == 0] = 1 # prevent division by zero
        E = E / norm
        return E

    def bps(self, x, a, reduce = True): 
        """
        Compute the negative log-likelihood (log base 2) of a sequence under the current model.

        Args:
            x (torch.Tensor): observation sequence (1D, int64)
            a (torch.Tensor): action sequence (1D, int64)
            reduce (bool): if True, return total log-likelihood. If False, return per-step values

        Returns:
            torch.Tensor: scalar if reduce=True, else [T] vector of per-step -log2 likelihoods
        """
        validate_seq(x, a, self.n_clones)

        x, a = x.to(self.device), a.to(self.device)

        log2_lik, _ = forward(
            self.T.transpose(0, 2, 1), 
            self.Pi_x, 
            self.n_clones, 
            x, a, self.device,
            store_messages = False
        )
        return -log2_lik.sum() if reduce else -log2_lik
    
    def bpsE(self, E, x, a, reduce = True):
        """
        Compute the negative log2-likelihood of a sequence under the current model with emissions.

        Args:
            E (torch.Tensor): [n_states, n_observations] Emission matrix
            x (torch.Tensor): [T] Observation sequence (int64)
            a (torch.Tensor): [T] Action sequence (int64)
            reduce (bool): If True, return total log-likelihood. If False, return per-step values

        Returns:
            torch.Tensor: scalar if reduce=True, else [T] vector
        """
        validate_seq(x, a, self.n_clones)
        x, a = x.to(self.device), a.to(self.device)
        E = E.to(self.device, dtype = self.dtype)

        log2_lik, _ = forwardE(
            self.T.transpose(0, 2, 1),
            E, 
            self.Pi_x,
            self.n_clones,
            x, a, self.device,
            store_messages = False
        )

        return -log2_lik.sum() if reduce else -log2_lik

    def bpsV(self, x, a, reduce = True):
        """
        Compute the negative log2-likelihood of a sequence under the current model
        using max-product (Viterbi) forward pass.

        This method finds the most likely clone trajectory (rather than computing marginals).

        Args:
            x (torch.Tensor): [T] observation sequence (int64)
            a (torch.Tensor): [T] action sequence (int64)
            reduce (bool): If True, returns scalar. Else returns per-step log-likelihoods.

        Returns:
            torch.Tensor: scalar if reduce=True, else [T] vector of per-step -log2 likelihoods.
        """
        validate_seq(x, a, self.n_clones)
        x, a = x.to(self.device), a.to(self.device)
        
        log2_lik, _ = forward_mp(
            self.T.transpose(0, 2, 1),
            self.Pi_x, 
            self.n_clones,
            x, a, self.device,
            store_messages = False
        )

        return -log2_lik.sum() if reduce else -log2_lik

    def decode(self, x, a):
        """
        Compute the Mean Average Precision (MAP) assignment of latent variables using max-product message passing.

        Args:
            x (torch.Tensor): [T] observation sequence (int64)
            a (torch.Tensor): [T] action sequence (int64)

        Returns:
            torch.Tensor: scalar -log2-likelihood of the MAP assignment
            torch.Tensor: [T] MAP assignment of latent variables
        """
        x, a = x.to(self.device), a.to(self.device)
        log2_lik, mess_fwd = forward_mp(
            self.T.transpose(0, 2, 1),
            self.Pi_x,
            self.n_clones,
            x, a, self.device,
            store_messages = True
        )
        states = backtrace(self.T, self.n_clones, x, a, mess_fwd, self.device)
        return -log2_lik, states
    
    def decodeE(self, E, x, a):
        """
        Compute the Mean Average Precision (MAP) assignment of latent variables using 
        max-product message passing with an alternative emission matrix

        Args:
            E (torch.Tensor): [n_states, n_observations] Emission matrix
            x (torch.Tensor): [T] Observation sequence (int64)
            a (torch.Tensor): [T] Action sequence (int64)

        Returns:
            torch.Tensor: scalar -log2-likelihood of the MAP assignment
            torch.Tensor: [T] MAP assignment of latent variables
        """
        x, a = x.to(self.device), a.to(self.device)
        E = E.to(self.device, dtype = self.dtype)

        log2_lik, mess_fwd = forwardE_mp(
            self.T.transpose(0, 2, 1), 
            E, 
            self.Pi_x,
            self.n_clones, 
            x, a, self.device,
            store_messages = True
        )
        states = backtraceE(self.T, E, self.n_clones, x, a, mess_fwd, self.device)
        return -log2_lik, states

    def learn_em_T(self, x, a, n_iter=100, term_early=True):
        """
        Run EM training, keeping E fixed and learning T from soft counts.

        Args:
            x (torch.Tensor): [T] Observation sequence (int64)
            a (torch.Tensor): [T] Action sequence (int64)
            n_iter (int): Number of EM iterations
            term_early (bool): If True, stop if no improvement in likelihood

        Returns:
            list[float]: Convergence history of negative log2-likelihood per step (BPS)
        """
        sys.stdout.flush()
        x, a = x.to(self.device), a.to(self.device)
        convergence = []

        pbar = trange(n_iter, position=0)
        log2_lik_old = -torch.inf

        for it in pbar:
            # === E-step: Compute soft transition expectations ===
            log2_lik, mess_fwd = forward(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x, a,
                store_messages=True
            )
            mess_bwd = backward(self.T, self.n_clones, x, a, self.device)
            updateC(self.C, self.T, self.n_clones, mess_fwd, mess_bwd, x, a, self.device)

            # === M-step: Normalize C into new T ===
            self.update_T()

            # === Convergence tracking ===
            bps = -log2_lik.mean()
            convergence.append(bps)
            pbar.set_postfix(train_bps=bps.item())

            if bps >= -log2_lik_old and term_early:
                break
            log2_lik_old = -bps

        return convergence

    def learn_viterbi_T(self, x, a, n_iter=100):
        """
        Run Viterbi training (hard EM) with a fixed emission matrix E.
        Updates the transition matrix T using the most likely clone assignments.

        Args:
            x (Tensor): [T] observation sequence
            a (Tensor): [T] action sequence
            n_iter (int): number of Viterbi EM iterations

        Returns:
            list[float]: convergence history (bits-per-step)
        """
        sys.stdout.flush()
        x, a = x.to(self.device), a.to(self.device)
        convergence = []

        pbar = trange(n_iter, position=0)
        bps_old = -torch.inf

        for it in pbar:
            # === E-step: Most likely clone trajectory ===
            log2_lik, mess_fwd = forward_mp(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x, a,
                self.device,
                store_messages=True
            )
            states = backtrace(self.T, self.n_clones, x, a, mess_fwd, self.device)

            # === Accumulate hard counts (GPU-optimized) ===
            self.C.zero_()
            with torch.no_grad():
                # Vectorized count accumulation
                if len(x) > 1:
                    t_indices = torch.arange(1, len(x), device=self.device)
                    a_indices = a[:-1]  # actions from t-1
                    i_indices = states[:-1]  # from states
                    j_indices = states[1:]  # to states
                    
                    # Use advanced indexing for parallel updates
                    self.C.index_put_((a_indices, i_indices, j_indices), 
                                     torch.ones_like(a_indices, dtype=self.dtype), 
                                     accumulate=True)

            # === M-step: Normalize counts into transition probabilities ===
            self.update_T()

            # === Convergence check ===
            bps = -log2_lik.mean()
            convergence.append(bps)
            pbar.set_postfix(train_bps=bps.item())

            if bps >= bps_old:
                break
            bps_old = bps

        return convergence

    def learn_em_E(self, x, a, n_iter=100, pseudocount_extra=1e-20):
        """
        Run soft EM training to learn the emission matrix E, while keeping transitions T fixed.

        Args:
            x (Tensor): [T] Observation sequence (int64)
            a (Tensor): [T] Action sequence (int64)
            n_iter (int): Number of EM iterations
            pseudocount_extra (float): Added for numerical stability in CE

        Returns:
            tuple:
                convergence (list[float]): Negative log-likelihood per iteration (BPS)
                E (Tensor): Final learned emission matrix [n_states, n_obs]
        """
        sys.stdout.flush()
        x, a = x.to(self.device), a.to(self.device)

        n_obs, n_states = len(self.n_clones), self.n_clones.sum()
        CE = torch.ones((n_states, n_obs), dtype=self.dtype, device=self.device)

        # Initialize E with uniform+small pseudocounts
        E = self.update_E(CE + pseudocount_extra)

        convergence = []
        pbar = trange(n_iter, position=0)
        bps_old = -torch.inf

        for it in pbar:
            # === E-step: Compute expected clone-observation alignment ===
            log2_lik, mess_fwd = forwardE(
                T_tr=self.T.transpose(0, 2, 1),
                E=E,
                Pi=self.Pi_x,
                n_clones=self.n_clones,
                x=x,
                a=a,
                device=self.device,
                store_messages=True,
            )
            mess_bwd = backwardE(self.T, E, self.n_clones, x, a, self.device)
            updateCE(CE, E, self.n_clones, mess_fwd, mess_bwd, x, a, self.device)

            # === M-step: Normalize CE into new E ===
            E = self.update_E(CE + pseudocount_extra)

            # === Convergence tracking ===
            bps = -log2_lik.mean()
            convergence.append(bps)
            pbar.set_postfix(train_bps=bps.item())

            if bps >= bps_old:
                break
            bps_old = bps

        return convergence, E

    def sample(self, length):
        """
        Sample an observation and action sequence from the CHMM.

        Args:
            length (int): Sequence length to generate

        Returns:
            sample_x (Tensor): [length] sampled observations (int64)
            sample_a (Tensor): [length] sampled actions (int64)
        """
        assert length > 0
        device = self.device
        n_actions = self.Pi_a.shape[0]
        state_loc = torch.cat([torch.tensor([0], device=device), self.n_clones.cumsum(0)])

        sample_x = torch.empty(length, dtype=torch.int64, device=device)
        sample_a = torch.multinomial(self.Pi_a, num_samples=length, replacement=True)

        # Sample initial clone from Pi_x
        p_h = self.Pi_x
        h = torch.multinomial(p_h, 1).item()

        for t in range(length):
            # Observation = which symbol does h belong to?
            obs = torch.searchsorted(state_loc, torch.tensor(h, device=device), right=False).item() - 1
            sample_x[t] = obs

            # Sample action
            a = sample_a[t]
            p_h = self.T[a, h]
            h = torch.multinomial(p_h, 1).item()

        return sample_x, sample_a
    
    def sample_sym(self, sym, length):
        """
        Sample a sequence of observations from the CHMM, conditioned on starting with symbol `sym`.

        Args:
            sym (int): Initial observation token
            length (int): Number of additional steps to generate

        Returns:
            list[int]: Observation token sequence of length `length+1`
        """
        assert length > 0
        device = self.device
        state_loc = torch.cat([torch.tensor([0], device=device), self.n_clones.cumsum(0)])

        seq = [sym]
        # Uniform distribution over clones of the starting symbol
        alpha = torch.ones(self.n_clones[sym], dtype=self.dtype, device=device)
        alpha /= alpha.sum()

        for _ in range(length):
            obs_tm1 = seq[-1]
            start, stop = state_loc[obs_tm1:obs_tm1 + 2]
            T_weighted = self.T.sum(dim=0)  # [from, to]

            # Extend alpha to full clone space
            long_alpha = alpha @ T_weighted[start:stop]  # shape: [n_states]
            long_alpha /= long_alpha.sum()

            idx = torch.multinomial(long_alpha, 1).item()
            sym = torch.searchsorted(state_loc, torch.tensor(idx, device=device), right=False).item() - 1
            seq.append(sym)

            new_start, new_stop = state_loc[sym:sym + 2]
            alpha = long_alpha[new_start:new_stop]
            alpha /= alpha.sum()

        return seq
    
    def bridge(self, state1, state2, max_steps=100):
        """
        Compute a likely action-observation trajectory from clone state1 to state2.

        Args:
            state1 (int): Starting clone state index
            state2 (int): Target clone state index
            max_steps (int): Maximum allowed steps for path

        Returns:
            list[Tuple[int, int]]: Sequence of (observation, action) pairs
        """
        Pi_x = torch.zeros(self.n_clones.sum(), dtype=self.dtype, device=self.device)
        Pi_x[state1] = 1.0

        log2_lik, mess_fwd = forward_mp_all(
            self.T.transpose(0, 2, 1),
            Pi_x,
            self.Pi_a,
            self.n_clones,
            state2,
            max_steps,
            device=self.device
        )
        s_a = backtrace_all(
            self.T,
            self.Pi_a,
            self.n_clones,
            mess_fwd,
            state2,
            device=self.device
        )
        return s_a

