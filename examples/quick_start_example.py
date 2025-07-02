#!/usr/bin/env python3
"""
CSCG-Torch Quick Start Example

A simple example demonstrating the core functionality of CSCG-Torch
with GPU-accelerated sequence generation and CHMM training.
"""

import time
import cscg_torch

def main():
    print("🚀 CSCG-Torch Quick Start Example")
    print("=" * 50)
    
    # 1. Check GPU setup
    print("\n1️⃣  GPU Detection and Optimization")
    device = cscg_torch.detect_optimal_device()
    gpu_info = cscg_torch.get_gpu_info(device)
    
    print(f"   Device: {gpu_info['name']}")
    print(f"   Memory: {gpu_info.get('memory_gb', 'N/A')} GB")
    print(f"   Optimizations: {', '.join(gpu_info['optimizations'])}")
    
    # 2. Load room data
    print("\n2️⃣  Loading Room Environment")
    try:
        room_data = cscg_torch.load_room_data("room_20x20")
        print(f"   Loaded room: {room_data.shape}")
        
        # Get room info
        room_info = cscg_torch.room_info(room_data)
        print(f"   Free cells: {room_info['free_cells']:,}")
        print(f"   Unique observations: {room_info['unique_observations']}")
        
    except FileNotFoundError:
        print("   No pre-generated room found, creating random room...")
        room_data = cscg_torch.create_random_room(20, 20, seed=42)
        print(f"   Created random room: {room_data.shape}")
    
    # 3. Create environment adapter
    print("\n3️⃣  Setting Up Environment Adapter")
    adapter = cscg_torch.create_room_adapter(room_data, adapter_type="torch", seed=42)
    print("   ✅ Room adapter created successfully")
    
    # 4. GPU-accelerated sequence generation
    print("\n4️⃣  GPU-Accelerated Sequence Generation")
    seq_length = 50_000
    print(f"   Generating {seq_length:,} steps...")
    
    start_time = time.time()
    x_seq, a_seq = adapter.generate_sequence_gpu(seq_length, device=device)
    generation_time = time.time() - start_time
    
    rate = seq_length / generation_time
    print(f"   ✅ Generated in {generation_time:.2f}s ({rate:,.0f} steps/sec)")
    print(f"   Obs range: [{x_seq.min()}, {x_seq.max()}]")
    print(f"   Action range: [{a_seq.min()}, {a_seq.max()}]")
    
    # 5. CHMM training
    print("\n5️⃣  CHMM Training with GPU Optimization")
    n_clones = cscg_torch.get_room_n_clones(n_clones_per_obs=100, device=device)
    total_states = n_clones.sum().item()
    print(f"   Training with {total_states:,} total states...")
    
    # Get GPU optimization settings
    gpu_settings = cscg_torch.optimize_for_gpu(device)
    
    start_time = time.time()
    model, progression = cscg_torch.train_chmm(
        n_clones=n_clones,
        x=x_seq,
        a=a_seq,
        device=device,
        method='em_T',
        n_iter=25,
        enable_mixed_precision=gpu_settings['mixed_precision'],
        early_stopping=True,
        seed=42
    )
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.1f}s")
    print(f"   Final BPS: {progression[-1]:.4f}")
    print(f"   Improvement: {progression[0] - progression[-1]:.4f} BPS")
    print(f"   Iterations: {len(progression)}")
    
    # 6. Model evaluation
    print("\n6️⃣  Model Evaluation")
    import torch
    
    x_tensor = torch.tensor(x_seq, device=device, dtype=torch.int64)
    a_tensor = torch.tensor(a_seq, device=device, dtype=torch.int64)
    
    # Calculate final performance
    final_bps = model.bps(x_tensor, a_tensor, reduce=True)
    print(f"   Final bits per step: {final_bps:.4f}")
    
    # Decode optimal sequence
    neg_log_lik, states = model.decode(x_tensor, a_tensor)
    unique_states = torch.unique(states).numel()
    print(f"   States used: {unique_states}/{total_states} ({unique_states/total_states*100:.1f}%)")
    
    # 7. Performance summary
    print("\n🎉 Success! Performance Summary:")
    print(f"   • GPU: {gpu_info['name']}")
    print(f"   • Sequence generation: {rate:,.0f} steps/sec")
    print(f"   • Training: {training_time:.1f}s for {len(progression)} iterations")
    print(f"   • Model performance: {final_bps:.4f} BPS")
    print(f"   • Memory usage: {cscg_torch.get_memory_info(device)['used_gb']:.1f} GB")
    
    print(f"\n✨ CSCG-Torch quick start completed successfully!")
    
    return model, progression, gpu_info

if __name__ == "__main__":
    try:
        model, progression, gpu_info = main()
        print("\n🚀 Ready for your research!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your installation and GPU setup.")
        raise