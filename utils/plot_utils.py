"""
Plotting and visualization utilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Union, Optional, List, Tuple

def plot_training_progression(progression: Union[List, np.ndarray, torch.Tensor],
                            title: str = "Training Progression",
                            xlabel: str = "Iteration", 
                            ylabel: str = "Bits Per Step (BPS)",
                            figsize: Tuple[int, int] = (10, 6),
                            save_path: Optional[str] = None,
                            show_improvement: bool = True) -> plt.Figure:
    """
    Plot training progression (BPS over iterations).
    
    Args:
        progression: List or array of BPS values over iterations
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        save_path: Path to save plot (optional)
        show_improvement: Whether to show improvement annotation
        
    Returns:
        matplotlib Figure object
        
    Examples:
        >>> fig = plot_training_progression(bps_values)
        >>> fig = plot_training_progression(progression, save_path="training.png")
    """
    # Convert to numpy array
    if isinstance(progression, torch.Tensor):
        prog_np = progression.cpu().numpy()
    else:
        prog_np = np.asarray(progression)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot progression
    iterations = np.arange(len(prog_np))
    ax.plot(iterations, prog_np, 'b-', linewidth=2, marker='o', 
            markersize=4, alpha=0.8, label='BPS')
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add improvement annotation
    if show_improvement and len(prog_np) > 1:
        improvement = prog_np[0] - prog_np[-1]
        improvement_pct = 100 * improvement / prog_np[0] if prog_np[0] != 0 else 0
        
        ax.text(0.02, 0.98, 
               f'Total Improvement: {improvement:.4f} BPS\n'
               f'Improvement: {improvement_pct:.2f}%', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training progression plot saved to {save_path}")
    
    return fig

def plot_room_layout(room_data: Union[np.ndarray, torch.Tensor],
                    title: str = "Room Layout",
                    figsize: Tuple[int, int] = (8, 6),
                    colormap: str = 'tab20',
                    show_colorbar: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot room layout with color-coded observations.
    
    Args:
        room_data: Room layout data
        title: Plot title
        figsize: Figure size (width, height)
        colormap: Colormap name for observations
        show_colorbar: Whether to show colorbar
        save_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
        
    Examples:
        >>> fig = plot_room_layout(room_array)
        >>> fig = plot_room_layout(room_tensor, colormap='viridis')
    """
    # Convert to numpy
    if isinstance(room_data, torch.Tensor):
        room_np = room_data.cpu().numpy()
    else:
        room_np = np.asarray(room_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap
    n_obs = int(room_np.max()) + 1 if room_np.size > 0 else 16
    cmap = plt.colormaps.get_cmap(colormap).resampled(n_obs)
    
    # Plot room
    im = ax.imshow(room_np, cmap=cmap, interpolation='nearest')
    
    # Add colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Observation Type', rotation=270, labelpad=15)
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X position", fontsize=12)
    ax.set_ylabel("Y position", fontsize=12)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Room layout plot saved to {save_path}")
    
    return fig

def plot_sequence_statistics(x_seq: Union[np.ndarray, torch.Tensor],
                           a_seq: Union[np.ndarray, torch.Tensor],
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot statistics of observation and action sequences.
    
    Args:
        x_seq: Observation sequence
        a_seq: Action sequence
        figsize: Figure size (width, height)
        save_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
        
    Examples:
        >>> fig = plot_sequence_statistics(observations, actions)
    """
    # Convert to numpy
    if isinstance(x_seq, torch.Tensor):
        x_np = x_seq.cpu().numpy()
    else:
        x_np = np.asarray(x_seq)
        
    if isinstance(a_seq, torch.Tensor):
        a_np = a_seq.cpu().numpy()
    else:
        a_np = np.asarray(a_seq)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Observation distribution
    unique_obs, obs_counts = np.unique(x_np, return_counts=True)
    ax1.bar(unique_obs, obs_counts, color='steelblue', alpha=0.7)
    ax1.set_title('Observation Distribution', fontweight='bold')
    ax1.set_xlabel('Observation ID')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Action distribution
    unique_acts, act_counts = np.unique(a_np, return_counts=True)
    ax2.bar(unique_acts, act_counts, color='forestgreen', alpha=0.7)
    ax2.set_title('Action Distribution', fontweight='bold')
    ax2.set_xlabel('Action ID')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Observation sequence over time (sample)
    sample_len = min(1000, len(x_np))
    sample_indices = np.linspace(0, len(x_np)-1, sample_len, dtype=int)
    ax3.plot(sample_indices, x_np[sample_indices], 'b-', alpha=0.7, linewidth=1)
    ax3.set_title('Observation Sequence (Sample)', fontweight='bold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Observation')
    ax3.grid(True, alpha=0.3)
    
    # Action sequence over time (sample)
    ax4.plot(sample_indices, a_np[sample_indices], 'g-', alpha=0.7, linewidth=1)
    ax4.set_title('Action Sequence (Sample)', fontweight='bold')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Action')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sequence statistics plot saved to {save_path}")
    
    return fig

def plot_gpu_performance(sequence_lengths: List[int],
                        generation_times: List[float],
                        device_name: str = "GPU",
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot GPU performance scaling with sequence length.
    
    Args:
        sequence_lengths: List of sequence lengths tested
        generation_times: List of corresponding generation times
        device_name: Name of the device
        figsize: Figure size (width, height)
        save_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
        
    Examples:
        >>> fig = plot_gpu_performance([1000, 10000, 100000], [0.1, 0.5, 2.0])
    """
    seq_lens = np.array(sequence_lengths)
    times = np.array(generation_times)
    rates = seq_lens / times  # steps per second
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Generation time vs sequence length
    ax1.loglog(seq_lens, times, 'bo-', linewidth=2, markersize=6)
    ax1.set_title(f'{device_name} Generation Time', fontweight='bold')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Generation Time (s)')
    ax1.grid(True, alpha=0.3)
    
    # Generation rate vs sequence length
    ax2.semilogx(seq_lens, rates, 'ro-', linewidth=2, markersize=6)
    ax2.set_title(f'{device_name} Generation Rate', fontweight='bold')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Generation Rate (steps/s)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GPU performance plot saved to {save_path}")
    
    return fig

def create_comparison_plot(data_dict: dict,
                          title: str = "Comparison",
                          xlabel: str = "X",
                          ylabel: str = "Y", 
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comparison plot with multiple data series.
    
    Args:
        data_dict: Dictionary mapping labels to (x, y) data tuples
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        save_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
        
    Examples:
        >>> data = {'CPU': (x1, y1), 'GPU': (x2, y2)}
        >>> fig = create_comparison_plot(data, "CPU vs GPU Performance")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (label, (x_data, y_data)) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(x_data, y_data, color=color, marker=marker, 
               linewidth=2, markersize=6, label=label, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    return fig