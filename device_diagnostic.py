#!/usr/bin/env python3
"""
Device transfer diagnostic script to identify CPU/GPU transfer bottlenecks.
"""

import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze')

def check_device_alignment():
    """Detailed device alignment check."""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Target device: {device}")
    
    # Create test data
    print("\n=== Creating test data ===")
    n_clones = torch.tensor([16] * 25, dtype=torch.int64, device=device)
    x = torch.randint(0, 25, (1000,), dtype=torch.int64, device=device)
    a = torch.randint(0, 4, (1000,), dtype=torch.int64, device=device)
    
    print(f"n_clones device: {n_clones.device}")
    print(f"x device: {x.device}")
    print(f"a device: {a.device}")
    
    # Initialize CHMM model
    print("\n=== Initializing CHMM model ===")
    from models.chmm_torch import CHMM_torch
    
    model = CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
    
    print(f"Model device: {model.device}")
    print(f"Model T device: {model.T.device}")
    print(f"Model Pi_x device: {model.Pi_x.device}")
    print(f"Model n_clones device: {model.n_clones.device}")
    
    # Check if there are any device mismatches
    all_devices = [
        ("n_clones_input", n_clones.device),
        ("x_input", x.device), 
        ("a_input", a.device),
        ("model.device", model.device),
        ("model.T", model.T.device),
        ("model.Pi_x", model.Pi_x.device),
        ("model.n_clones", model.n_clones.device),
        ("target", device)
    ]
    
    print("\n=== Device Summary ===")
    device_issues = []
    for name, tensor_device in all_devices:
        status = "✅" if tensor_device == device else "❌"
        print(f"{status} {name}: {tensor_device}")
        if tensor_device != device:
            device_issues.append(name)
    
    if device_issues:
        print(f"\n❌ Device mismatches found in: {', '.join(device_issues)}")
        return False
    else:
        print(f"\n✅ All tensors properly aligned to {device}")
        return True

def trace_device_transfers_in_forward():
    """Trace device transfers during forward pass."""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"\n=== Tracing forward pass device transfers ===")
    
    # Create aligned test data
    n_clones = torch.tensor([16] * 25, dtype=torch.int64, device=device)
    x = torch.randint(0, 25, (100,), dtype=torch.int64, device=device)
    a = torch.randint(0, 4, (100,), dtype=torch.int64, device=device)
    
    from models.chmm_torch import CHMM_torch
    model = CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
    
    # Check model parameter devices before forward
    print("Before forward pass:")
    print(f"  T device: {model.T.device}")
    print(f"  Pi_x device: {model.Pi_x.device}")
    print(f"  n_clones device: {model.n_clones.device}")
    print(f"  x device: {x.device}")
    print(f"  a device: {a.device}")
    
    # Run forward pass
    from models.train_utils import forward
    T_tr = model.T.transpose(1, 2)
    
    print(f"  T_tr device: {T_tr.device}")
    
    try:
        log2_lik, mess_fwd = forward(T_tr, model.Pi_x, model.n_clones, x, a, device, 
                                   store_messages=True, workspace={})
        print("✅ Forward pass completed")
        print(f"  Output log2_lik device: {log2_lik.device}")
        if mess_fwd is not None:
            print(f"  Output mess_fwd device: {mess_fwd.device}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    return True

def check_chmm_device_initialization():
    """Check device handling in CHMM_torch initialization."""
    print(f"\n=== Checking CHMM device initialization ===")
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # Test with explicit device placement
    n_clones = torch.tensor([16] * 25, dtype=torch.int64, device=device)
    x = torch.randint(0, 25, (100,), dtype=torch.int64, device=device)
    a = torch.randint(0, 4, (100,), dtype=torch.int64, device=device)
    
    print(f"Input tensors all on {device}")
    
    from models.chmm_torch import CHMM_torch
    
    # Check if CHMM_torch properly handles device in __init__
    try:
        model = CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
        
        # Check internal device setup
        print(f"Model reports device: {model.device}")
        print(f"Model T matrix device: {model.T.device}")
        print(f"Model Pi_x device: {model.Pi_x.device}")
        
        # Check if CHMM correctly identified device from inputs
        if hasattr(model, '_setup_device_and_memory'):
            print("✅ CHMM has _setup_device_and_memory method")
        else:
            print("❌ CHMM missing device setup method")
            
        return model.device == device and model.T.device == device
        
    except Exception as e:
        print(f"❌ CHMM initialization failed: {e}")
        return False

def check_train_utils_device_handling():
    """Check device handling in train_utils functions."""
    print(f"\n=== Checking train_utils device handling ===")
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # Read the relevant functions and check for device transfer logic
    from models.train_utils import forward, backward, updateC
    
    # Test each function's device handling
    n_clones = torch.tensor([16] * 25, dtype=torch.int64, device=device)
    x = torch.randint(0, 25, (50,), dtype=torch.int64, device=device)
    a = torch.randint(0, 4, (50,), dtype=torch.int64, device=device)
    
    from models.chmm_torch import CHMM_torch
    model = CHMM_torch(n_clones=n_clones, x=x, a=a, pseudocount=0.01, seed=42)
    
    issues = []
    
    # Test forward
    try:
        T_tr = model.T.transpose(1, 2)
        log2_lik, mess_fwd = forward(T_tr, model.Pi_x, model.n_clones, x, a, device)
        if log2_lik.device != device:
            issues.append(f"forward: output on {log2_lik.device}, expected {device}")
    except Exception as e:
        issues.append(f"forward: {e}")
    
    # Test backward 
    try:
        mess_bwd = backward(model.T, model.n_clones, x, a, device)
        if mess_bwd.device != device:
            issues.append(f"backward: output on {mess_bwd.device}, expected {device}")
    except Exception as e:
        issues.append(f"backward: {e}")
    
    if issues:
        print("❌ Device handling issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ train_utils functions handle devices correctly")
        return True

def main():
    """Main diagnostic function."""
    print("CHMM Device Transfer Diagnostic")
    print("=" * 50)
    
    # Check basic device alignment
    basic_ok = check_device_alignment()
    
    # Check CHMM initialization device handling
    init_ok = check_chmm_device_initialization()
    
    # Check train_utils device handling
    utils_ok = check_train_utils_device_handling()
    
    # Trace device transfers in forward pass
    forward_ok = trace_device_transfers_in_forward()
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    print(f"Basic device alignment: {'✅' if basic_ok else '❌'}")
    print(f"CHMM initialization: {'✅' if init_ok else '❌'}")
    print(f"train_utils functions: {'✅' if utils_ok else '❌'}")
    print(f"Forward pass: {'✅' if forward_ok else '❌'}")
    
    if not (basic_ok and init_ok and utils_ok and forward_ok):
        print("\n🔥 CRITICAL: Device transfer issues detected!")
        print("This is likely the primary cause of slow A100 performance.")
        print("\nRecommended fixes:")
        
        if not basic_ok:
            print("1. Fix tensor device alignment in initialization")
        if not init_ok:
            print("2. Fix CHMM_torch device setup in __init__")
        if not utils_ok:
            print("3. Fix device handling in train_utils functions")
        if not forward_ok:
            print("4. Fix forward pass device management")
            
    else:
        print("\n✅ All device handling appears correct")
        print("Device transfers are not the bottleneck.")

if __name__ == "__main__":
    main()