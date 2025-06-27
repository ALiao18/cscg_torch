#!/usr/bin/env python3
"""
Integration Demo for Enhanced CSCG_torch Functions

This script demonstrates the integrated functionality of all new functions:
- Enhanced training with train_chmm
- Improved plotting with enhanced labels and legends  
- GPU-optimized computation
- Room environment utilities
- Performance analysis
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# Import enhanced functions
from cscg_torch.models.train_utils import (
    train_chmm, make_E, make_E_sparse, compute_forward_messages, place_field
)
from cscg_torch.env_adapters.room_utils import (
    demo_room_setup, clone_to_obs_map, top_k_used_clones, count_used_clones
)
from cscg_torch.env_adapters.room_adapter import save_room_plot
from cscg_torch.env_adapters.base_adapter import plot_graph


def main():
    """Run the complete integration demonstration."""
    print("CSCG_torch Integration Demo")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    demo_dir = Path("demo_plots")
    demo_dir.mkdir(exist_ok=True)
    
    # Stage 1: Environment Setup
    print("\n1. Setting up room environment...")
    adapter, n_clones, (x_seq, a_seq) = demo_room_setup()
    
    # Convert to tensors and move to device
    x = torch.tensor(x_seq, device=device)
    a = torch.tensor(a_seq, device=device) 
    n_clones = n_clones.to(device)
    
    print(f"   Room size: {adapter.h}x{adapter.w}")
    print(f"   Sequence length: {len(x)}")
    print(f"   Number of clones: {n_clones.sum().item()}")
    print(f"   Observation types: {len(n_clones)}")
    
    # Stage 2: Model Training
    print("\n2. Training CHMM models...")
    
    methods = ['em_T', 'viterbi_T']
    trained_models = {}
    progressions = {}
    
    for method in methods:
        print(f"   Training with {method}...")
        start_time = time.time()
        
        model, progression = train_chmm(
            n_clones, x, a, 
            device=device,
            method=method,
            n_iter=20,
            pseudocount=0.01,
            early_stopping=True
        )
        
        training_time = time.time() - start_time
        trained_models[method] = model
        progressions[method] = progression
        
        final_bps = model.bps(x, a, reduce=True).item()
        print(f"     Training time: {training_time:.2f}s")
        print(f"     Final BPS: {final_bps:.4f}")
        print(f"     Iterations: {len(progression)}")
    
    # Stage 3: Analysis and Visualization
    print("\n3. Computing forward messages and place fields...")
    
    # Use the best model (em_T)
    best_model = trained_models['em_T']
    
    # Create emission matrix
    E = make_E(n_clones, device)
    print(f"   Emission matrix shape: {E.shape}")
    
    # Compute forward messages
    chmm_state = {
        'T': best_model.T,
        'E': E,
        'Pi_x': best_model.Pi_x, 
        'n_clones': n_clones
    }
    
    mess_fwd = compute_forward_messages(chmm_state, x, a, device)
    print(f"   Forward messages shape: {mess_fwd.shape}")
    
    # Generate position data for place fields
    T_len = len(x)
    rc = torch.randint(0, 5, (T_len, 2), device=device)  # Mock 5x5 room positions
    
    # Compute place fields for top clones
    states = best_model.decode(x, a)[1]
    top_clones = top_k_used_clones(states.cpu().numpy(), k=5)
    print(f"   Top 5 clones: {top_clones}")
    
    # Stage 4: Enhanced Plotting
    print("\n4. Creating enhanced plots...")
    
    # Plot training progression
    plot_graph(
        best_model,
        progression=progressions['em_T'],
        plot_mode='progression',
        trial_name='demo',
        save_format='both'
    )
    
    # Plot usage analysis
    plot_graph(
        best_model,
        x=x, a=a,
        plot_mode='usage',
        trial_name='demo',
        save_format='both'
    )
    
    # Plot performance analysis
    plot_graph(
        best_model,
        x=x, a=a,
        plot_mode='performance', 
        trial_name='demo',
        save_format='both'
    )
    
    # Save room layout plot
    room_array = adapter.room.cpu().numpy() if hasattr(adapter.room, 'cpu') else adapter.room
    save_room_plot(
        room_array,
        demo_dir / "room_layout",
        title="Demo Room Environment",
        show_grid=True,
        save_formats=['pdf', 'png']
    )
    
    # Stage 5: Performance Analysis
    print("\n5. Performance analysis...")
    
    # Clone usage analysis
    usage_counts = count_used_clones(best_model, x, a)
    total_unique_clones = sum(usage_counts.values())
    max_possible_clones = n_clones.sum().item()
    
    print(f"   Clone usage by observation type:")
    for obs_id, count in usage_counts.items():
        print(f"     Obs {obs_id}: {count}/{n_clones[obs_id].item()} clones used")
    
    print(f"   Total unique clones used: {total_unique_clones}/{max_possible_clones}")
    print(f"   Clone utilization: {(total_unique_clones/max_possible_clones)*100:.1f}%")
    
    # Compare training methods
    print(f"\n6. Training method comparison:")
    for method, progression in progressions.items():
        initial_bps = progression[0]
        final_bps = progression[-1]
        improvement = initial_bps - final_bps
        print(f"   {method}:")
        print(f"     Initial BPS: {initial_bps:.4f}")
        print(f"     Final BPS: {final_bps:.4f}")
        print(f"     Improvement: {improvement:.4f}")
        print(f"     Iterations: {len(progression)}")
    
    # Stage 6: GPU Performance Metrics (if available)
    if device.type == 'cuda':
        print(f"\n7. GPU Performance metrics:")
        
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
        
        print(f"   GPU: {torch.cuda.get_device_name(device)}")
        print(f"   Memory allocated: {memory_allocated:.1f} MB") 
        print(f"   Memory reserved: {memory_reserved:.1f} MB")
        
        # Test GPU-optimized place field computation
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for i in range(min(5, len(top_clones))):
            clone_id = top_clones[i][0]
            if clone_id < mess_fwd.shape[1]:
                field = place_field(mess_fwd, rc, clone_id, device)
        end_event.record()
        
        torch.cuda.synchronize()
        place_field_time = start_event.elapsed_time(end_event)
        print(f"   Place field computation time: {place_field_time:.2f}ms")
    
    # Stage 7: Summary
    print(f"\n8. Demo Summary:")
    print(f"   ✓ Successfully trained {len(methods)} CHMM models")
    print(f"   ✓ Computed forward messages for {len(x)} time steps")
    print(f"   ✓ Generated enhanced plots with proper labels and legends")
    print(f"   ✓ Analyzed clone usage across {len(n_clones)} observation types")
    print(f"   ✓ Demonstrated GPU optimization (if available)")
    
    plot_files = list(demo_dir.glob("**/*"))
    print(f"   ✓ Created {len(plot_files)} output files in {demo_dir}")
    
    print(f"\nAll files saved to: {demo_dir.absolute()}")
    print(f"Demo completed successfully!")
    
    return {
        'trained_models': trained_models,
        'progressions': progressions,
        'usage_counts': usage_counts,
        'demo_dir': demo_dir
    }


def validate_improvements():
    """Validate that improvements are working correctly."""
    print("\nValidating improvements...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test 1: GPU compatibility
    n_clones = torch.tensor([2, 2, 2, 2], device=device)
    x = torch.randint(0, 4, (50,), device=device)
    a = torch.randint(0, 4, (50,), device=device)
    
    try:
        model, progression = train_chmm(n_clones, x, a, device=device, n_iter=2)
        assert model.device == device, "GPU compatibility failed"
        print("   ✓ GPU compatibility working")
    except Exception as e:
        print(f"   ✗ GPU compatibility failed: {e}")
    
    # Test 2: Enhanced emission matrix functions
    try:
        E_dense = make_E(n_clones, device)
        E_sparse = make_E_sparse(n_clones, device)
        assert torch.allclose(E_dense, E_sparse.to_dense()), "Emission matrix consistency failed"
        print("   ✓ Emission matrix functions working")
    except Exception as e:
        print(f"   ✗ Emission matrix functions failed: {e}")
    
    # Test 3: Place field GPU optimization
    try:
        mess_fwd = torch.rand(50, n_clones.sum().item(), device=device)
        rc = torch.randint(0, 5, (50, 2), device=device)
        field = place_field(mess_fwd, rc, 0, device)
        assert field.device == device, "Place field GPU optimization failed"
        print("   ✓ Place field GPU optimization working")
    except Exception as e:
        print(f"   ✗ Place field GPU optimization failed: {e}")
    
    # Test 4: Plotting enhancements
    try:
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        room = np.random.randint(-1, 5, (5, 5))
        
        save_room_plot(room, os.path.join(temp_dir, "test"), save_formats=['png'])
        
        # Check if file was created
        test_file = os.path.join(temp_dir, "test.png")
        if os.path.exists(test_file):
            print("   ✓ Enhanced plotting working")
            os.remove(test_file)
        else:
            print("   ✗ Enhanced plotting failed - file not created")
            
        os.rmdir(temp_dir)
        
    except Exception as e:
        print(f"   ✗ Enhanced plotting failed: {e}")
    
    print("Validation complete.")


if __name__ == "__main__":
    try:
        results = main()
        validate_improvements()
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU memory if used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()