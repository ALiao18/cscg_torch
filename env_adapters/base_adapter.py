"""
Base Environment Adapter

Abstract base class for environment adapters used with CHMM models.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    import igraph
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    igraph = None

# Default plot directory
PLOT_DIR = "plots"

class CSCGEnvironmentAdapter:
    def __init__(self, seed=42):
        # Strict type assertions for initialization
        assert isinstance(seed, int), f"seed must be int, got {type(seed)}"
        assert seed >= 0, f"seed must be non-negative, got {seed}"
        
        self.rng = np.random.RandomState(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_actions = None  # Should be set by subclasses
        
        # Post-initialization assertions
        assert isinstance(self.rng, np.random.RandomState), f"rng must be RandomState, got {type(self.rng)}"
        assert isinstance(self.device, torch.device), f"device must be torch.device, got {type(self.device)}"

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def is_terminal(self):
        return False

    def generate_sequence(self, length):
        # Strict input validation
        assert isinstance(length, int), f"length must be int, got {type(length)}"
        assert length > 0, f"length must be positive, got {length}"
        assert self.n_actions is not None, "n_actions must be set by subclass"
        assert isinstance(self.n_actions, int), f"n_actions must be int, got {type(self.n_actions)}"
        assert self.n_actions > 0, f"n_actions must be positive, got {self.n_actions}"
        
        x_seq, a_seq = [], []
        self.reset()
        
        for i in range(length):
            obs = self.get_observation()
            
            # Strict type checking for observation
            assert obs is not None, f"get_observation() returned None at step {i}"
            assert isinstance(obs, (int, np.integer)), f"obs must be int, got {type(obs)} at step {i}"
            
            action = self.rng.choice(self.n_actions)
            
            # Strict type checking for action
            assert isinstance(action, (int, np.integer)), f"action must be int, got {type(action)} at step {i}"
            assert 0 <= action < self.n_actions, f"action {action} out of range [0, {self.n_actions}) at step {i}"
            
            new_obs, valid = self.step(action)
            
            # Strict type checking for step results
            assert isinstance(valid, bool), f"step() must return bool valid, got {type(valid)} at step {i}"
            
            if valid:
                # Ensure types are safe for numpy array conversion
                x_seq.append(int(obs))
                a_seq.append(int(action))
        
        # Final output validation
        result_x = np.array(x_seq, dtype=np.int64)
        result_a = np.array(a_seq, dtype=np.int64)
        
        assert len(result_x) == len(result_a), f"Sequence lengths mismatch: x={len(result_x)}, a={len(result_a)}"
        assert len(result_x) <= length, f"Generated sequence too long: {len(result_x)} > {length}"
        
        return result_x, result_a
    
# Note: These functions are now defined in room_utils.py to avoid duplication
# Import them from room_utils when needed

def plot_graph(
    chmm,
    x=None,
    a=None,
    room=None,
    mess_fwd=None,
    rc=None,
    progression=None,
    plot_mode='cscg',
    trial_name='trial_default',
    k=20,
    save_format='pdf'
):
    """
    Create various plots for CHMM analysis with improved performance and labeling.
    
    Args:
        chmm: CHMM model
        x (array-like): Observation sequence
        a (array-like): Action sequence  
        room (array-like): Room layout for room plots
        mess_fwd (array-like): Forward messages for place field plots
        rc (array-like): Row-column positions for place field plots
        progression (array-like): Training progression data
        plot_mode (str): Type of plot ('cscg', 'room', 'place_fields', 'progression', 'usage', 'performance')
        trial_name (str): Name for trial directory
        k (int): Number of top clones for place field plots
        save_format (str): Format to save plots ('pdf', 'png', or 'both')
    """
    # Input validation
    assert chmm is not None, "chmm cannot be None"
    assert isinstance(plot_mode, str), f"plot_mode must be str, got {type(plot_mode)}"
    assert plot_mode in ['cscg', 'room', 'place_fields', 'progression', 'usage', 'performance'], f"invalid plot_mode: {plot_mode}"
    assert isinstance(trial_name, str), f"trial_name must be str, got {type(trial_name)}"
    assert isinstance(k, int), f"k must be int, got {type(k)}"
    assert k > 0, f"k must be positive, got {k}"
    
    # Import helper functions
    from .room_utils import clone_to_obs_map, top_k_used_clones, count_used_clones
    
    # Create trial folder
    trial_dir = os.path.join(PLOT_DIR, trial_name)
    os.makedirs(trial_dir, exist_ok=True)

    # Determine save format and paths
    if save_format == 'both':
        paths = [
            os.path.join(trial_dir, f"{plot_mode}.pdf"),
            os.path.join(trial_dir, f"{plot_mode}.png")
        ]
    else:
        extension = 'pdf' if save_format == 'pdf' else 'png'
        paths = [os.path.join(trial_dir, f"{plot_mode}.{extension}")]
    
    full_path = paths[0]  # Primary path for backward compatibility

    if plot_mode == 'room':
        assert room is not None, "room required for room plot"
        
        # Convert to numpy if needed
        if isinstance(room, torch.Tensor):
            room = room.cpu().numpy()
        room = np.asarray(room)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        n_obs = int(room.max()) + 1
        cmap = plt.colormaps.get_cmap('tab20').resampled(n_obs)
        im = ax.imshow(room, cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Observation Type', rotation=270, labelpad=15)
        ax.set_title("Maze Observation Layout", fontsize=14, fontweight='bold')
        ax.set_xlabel("X position", fontsize=12)
        ax.set_ylabel("Y position", fontsize=12)
        plt.tight_layout()
        
        # Save in specified format(s)
        for path in paths:
            fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    elif plot_mode == 'cscg':
        assert x is not None and a is not None, "x and a required for cscg plot"
        
        if not IGRAPH_AVAILABLE:
            print("Warning: igraph not available, skipping CSCG plot")
            return
            
        # Convert sequences to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
            
        # Decode states
        _, states = chmm.decode(x, a)
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
            
        v = np.unique(states)
        
        # Get transition counts and normalize
        if hasattr(chmm, 'C'):
            T = chmm.C[:, v][:, :, v]
        else:
            # Fallback if C not available
            print("Warning: C matrix not available, using T matrix")
            T = chmm.T[:, v][:, :, v]
            
        A = T.sum(0)
        A = A / (A.sum(1, keepdims=True) + 1e-10)  # Add small epsilon to avoid division by zero

        g = igraph.Graph.Adjacency((A > 0).tolist(), mode=igraph.ADJ_DIRECTED)

        obs_map = clone_to_obs_map(chmm.n_clones)
        node_obs_ids = [obs_map[vi] for vi in v]
        cmap = plt.colormaps.get_cmap('tab20').resampled(len(chmm.n_clones))
        colors = [cmap(obs_id)[:3] for obs_id in node_obs_ids]

        freq_values = np.array([np.sum(states == vi) for vi in v])
        freq_values = np.clip(freq_values / (freq_values.max() + 1e-10), 0.2, 1.0)
        vertex_sizes = 30 * freq_values

        igraph.plot(
            g,
            target=full_path,
            layout=g.layout("kamada_kawai"),
            vertex_color=colors,
            vertex_size=vertex_sizes,
            vertex_label=[str(vi) for vi in v],
            margin=50,
            bbox=(800, 600)
        )

    elif plot_mode == 'place_fields':
        assert x is not None and a is not None, "x and a required for place fields plot"
        assert mess_fwd is not None, "mess_fwd required for place fields plot"
        assert rc is not None, "rc required for place fields plot"
        
        # Convert to numpy if needed
        if isinstance(mess_fwd, torch.Tensor):
            mess_fwd = mess_fwd.cpu().numpy()
        if isinstance(rc, torch.Tensor):
            rc = rc.cpu().numpy()
            
        # Decode states
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        _, states = chmm.decode(x, a)
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
            
        top_clones = top_k_used_clones(states, k)
        n_plots = min(k, 5, len(top_clones))
        
        fig, axs = plt.subplots(1, n_plots, figsize=(3*n_plots, 4))
        if n_plots == 1:
            axs = [axs]
            
        for i, (clone, _) in enumerate(top_clones[:n_plots]):
            ax = axs[i]
            pf = np.zeros(rc.max(0) + 1)
            count = np.zeros(rc.max(0) + 1, int)
            for t in range(mess_fwd.shape[0]):
                r, c = rc[t]
                pf[r, c] += mess_fwd[t, clone]
                count[r, c] += 1
            count[count == 0] = 1
            pf /= count
            im = ax.imshow(pf, cmap='viridis')
            ax.set_title(f"Clone {clone}")
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            
        fig.suptitle("Place Fields of Top Used Clones", fontsize=16, fontweight='bold')
        plt.tight_layout()
        for path in paths:
            fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    elif plot_mode == 'progression':
        assert progression is not None, "progression required for progression plot"
        
        # Convert to numpy if needed
        if isinstance(progression, torch.Tensor):
            progression = progression.cpu().numpy()
        progression = np.asarray(progression)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(progression, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8)
        ax.set_title("EM Training Progression", fontsize=14, fontweight='bold')
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Bits Per Step (BPS)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if len(progression) > 1:
            improvement = progression[0] - progression[-1]
            ax.text(0.02, 0.98, f'Total Improvement: {improvement:.4f} BPS', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        for path in paths:
            fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    elif plot_mode == 'usage':
        assert x is not None and a is not None, "x and a required for usage plot"
        
        usage = count_used_clones(chmm, x, a)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(list(usage.keys()), list(usage.values()), 
                     color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title("Clone Usage Per Observation Type", fontsize=14, fontweight='bold')
        ax.set_xlabel("Observation ID", fontsize=12)
        ax.set_ylabel("Number of Clones Used", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Add total usage info
        total_usage = sum(usage.values())
        ax.text(0.98, 0.98, f'Total Clones Used: {total_usage}', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        for path in paths:
            fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    elif plot_mode == 'performance':
        assert x is not None and a is not None, "x and a required for performance plot"
        
        # Import forward function
        try:
            from ..models.train_utils import forward
        except ImportError:
            from cscg_torch.models.train_utils import forward

        fig, ax = plt.subplots(figsize=(8, 6))

        # Convert to torch tensors if needed
        if isinstance(x, np.ndarray):
            x_torch = torch.from_numpy(x)
        else:
            x_torch = x
        if isinstance(a, np.ndarray):
            a_torch = torch.from_numpy(a)
        else:
            a_torch = a

        # Compute forward log2 likelihoods for the whole sequence
        device = next(chmm.parameters()).device
        log2_lik, _ = forward(
            chmm.T.transpose(0, 2, 1),
            chmm.Pi_x,
            chmm.n_clones,
            x_torch.to(device),
            a_torch.to(device),
            device,
            store_messages=False,
        )

        if isinstance(log2_lik, torch.Tensor):
            log2_lik = log2_lik.cpu().numpy()

        prob_next_obs = np.exp2(log2_lik)  # convert log2 prob to prob

        # Clone visitation: compute unique clone coverage over time
        _, states = chmm.decode(x_torch, a_torch)
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
            
        seen = set()
        prop_explored = []
        unique_states = np.unique(states)
        for s in states:
            seen.add(s)
            prop_explored.append(len(seen) / len(unique_states))

        ax.plot(prob_next_obs, label="Prob. next obs.", color='black', linewidth=2, alpha=0.8)
        ax.plot(prop_explored, label="Prop. explored nodes", color='red', linewidth=2, alpha=0.8)
        ax.set_xlabel("Number of actions", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        ax.set_title("Performance Analysis: Prediction vs. Exploration", fontsize=14, fontweight='bold')
        
        # Add final values as text
        final_prob = prob_next_obs[-1] if len(prob_next_obs) > 0 else 0
        final_explored = prop_explored[-1] if len(prop_explored) > 0 else 0
        ax.text(0.02, 0.02, f'Final Prob.: {final_prob:.3f}\nFinal Explored: {final_explored:.3f}', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        for path in paths:
            plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    else:
        raise ValueError(f"Unsupported plot_mode: {plot_mode}")
    
    # Print save confirmation
    if len(paths) == 1:
        print(f"Plot saved to {full_path}")
    else:
        print(f"Plot saved to {', '.join(paths)}")