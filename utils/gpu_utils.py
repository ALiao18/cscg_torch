"""
GPU optimization and device detection utilities.
"""

import torch
import platform
from typing import Dict, Any, Optional, Tuple

def detect_optimal_device(prefer_cuda: bool = True) -> torch.device:
    """
    Automatically detect the optimal compute device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA over MPS when both available
        
    Returns:
        torch.device: Optimal device for computation
        
    Examples:
        >>> device = detect_optimal_device()
        >>> print(device)  # cuda:0, mps:0, or cpu
    """
    if torch.cuda.is_available() and prefer_cuda:
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return torch.device("cpu")

def get_gpu_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get comprehensive GPU information.
    
    Args:
        device: Device to query (defaults to optimal device)
        
    Returns:
        Dictionary with GPU information
        
    Examples:
        >>> info = get_gpu_info()
        >>> print(f"GPU: {info['name']}, Memory: {info['memory_gb']:.1f}GB")
    """
    if device is None:
        device = detect_optimal_device()
    
    info = {
        'device': device,
        'device_type': device.type,
        'available': False,
        'name': 'Unknown',
        'memory_gb': 0.0,
        'compute_capability': None,
        'optimizations': []
    }
    
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(device)
            info.update({
                'available': True,
                'name': gpu_props.name,
                'memory_gb': gpu_props.total_memory / (1024**3),
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'multiprocessor_count': gpu_props.multi_processor_count
            })
            
            # Determine GPU-specific optimizations
            gpu_name = gpu_props.name
            if "V100" in gpu_name:
                info['optimizations'] = ['V100 Tensor Cores', 'FP16 Mixed Precision', 'Large Chunks']
            elif any(gpu in gpu_name for gpu in ["A100", "H100"]):
                info['optimizations'] = ['Advanced Tensor Cores', 'BF16/FP16 Precision', 'Very Large Chunks']
            elif "RTX" in gpu_name or "GTX" in gpu_name:
                info['optimizations'] = ['Consumer GPU Optimization', 'Standard Chunks']
            else:
                info['optimizations'] = ['General CUDA Optimization']
                
        except Exception as e:
            info['error'] = str(e)
            
    elif device.type == 'mps' and torch.backends.mps.is_available():
        info.update({
            'available': True,
            'name': 'Apple Silicon (MPS)',
            'memory_gb': 'Unified Memory',
            'optimizations': ['MPS Acceleration', 'Unified Memory', 'Small Chunks']
        })
        
    elif device.type == 'cpu':
        info.update({
            'available': True,
            'name': f"{platform.processor() or 'CPU'}",
            'optimizations': ['CPU Fallback', 'NumPy Operations']
        })
    
    return info

def optimize_for_gpu(device: torch.device) -> Dict[str, Any]:
    """
    Get optimization settings for specific GPU.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with optimization settings
        
    Examples:
        >>> settings = optimize_for_gpu(torch.device('cuda:0'))
        >>> print(settings['chunk_size'])
    """
    gpu_info = get_gpu_info(device)
    
    # Default settings
    settings = {
        'chunk_size': 1024,
        'mixed_precision': False,
        'tensor_cores': False,
        'memory_optimization': 'standard',
        'batch_size_multiplier': 1.0
    }
    
    if device.type == 'cuda' and gpu_info['available']:
        gpu_name = gpu_info['name']
        memory_gb = gpu_info.get('memory_gb', 0)
        
        # V100 optimizations
        if "V100" in gpu_name:
            settings.update({
                'chunk_size': 32768,
                'mixed_precision': True,
                'tensor_cores': True,
                'memory_optimization': 'v100',
                'batch_size_multiplier': 2.0
            })
            
        # A100/H100 optimizations
        elif any(gpu in gpu_name for gpu in ["A100", "H100"]):
            settings.update({
                'chunk_size': 65536,
                'mixed_precision': True,
                'tensor_cores': True,
                'memory_optimization': 'a100',
                'batch_size_multiplier': 4.0
            })
            
        # RTX series optimizations
        elif "RTX" in gpu_name:
            if "4090" in gpu_name or "4080" in gpu_name:
                settings.update({
                    'chunk_size': 24576,
                    'mixed_precision': True,
                    'tensor_cores': True,
                    'memory_optimization': 'rtx40',
                    'batch_size_multiplier': 1.5
                })
            else:
                settings.update({
                    'chunk_size': 16384,
                    'mixed_precision': True,
                    'tensor_cores': True,
                    'memory_optimization': 'rtx',
                    'batch_size_multiplier': 1.2
                })
                
        # Memory-based scaling for unknown GPUs
        else:
            memory_factor = min(memory_gb / 16.0, 2.0)  # Relative to V100's 16GB
            settings.update({
                'chunk_size': int(16384 * memory_factor),
                'mixed_precision': memory_gb >= 8,
                'memory_optimization': 'general',
                'batch_size_multiplier': memory_factor
            })
            
    elif device.type == 'mps':
        settings.update({
            'chunk_size': 8192,
            'mixed_precision': False,  # MPS has limited FP16 support
            'tensor_cores': False,
            'memory_optimization': 'mps',
            'batch_size_multiplier': 0.8
        })
        
    return settings

def benchmark_device(device: torch.device, 
                    test_sizes: Tuple[int, ...] = (1000, 10000),
                    warmup: bool = True) -> Dict[str, float]:
    """
    Benchmark device performance with matrix operations.
    
    Args:
        device: Device to benchmark
        test_sizes: Sizes to test
        warmup: Whether to run warmup iterations
        
    Returns:
        Dictionary with benchmark results (GFLOPS)
        
    Examples:
        >>> results = benchmark_device(torch.device('cuda:0'))
        >>> print(f"Performance: {results['gflops']:.1f} GFLOPS")
    """
    import time
    
    results = {
        'device': str(device),
        'gflops': 0.0,
        'memory_bandwidth_gb_s': 0.0,
        'error': None
    }
    
    try:
        # Warmup iterations
        if warmup:
            for _ in range(3):
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.matmul(a, b)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                del a, b, c
        
        # Actual benchmark
        total_gflops = 0.0
        num_tests = len(test_sizes)
        
        for size in test_sizes:
            # Create test matrices
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            # Time matrix multiplication
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            c = torch.matmul(a, b)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Calculate GFLOPS (2 * N^3 operations for NxN matrix multiplication)
            operations = 2 * size**3
            elapsed_time = end_time - start_time
            gflops = (operations / elapsed_time) / 1e9
            
            total_gflops += gflops
            
            del a, b, c
        
        results['gflops'] = total_gflops / num_tests
        
        # Memory bandwidth test
        if device.type in ['cuda', 'mps']:
            try:
                size = 10000000  # 10M elements
                data = torch.randn(size, device=device, dtype=torch.float32)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                result = data.sum()
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Calculate bandwidth (bytes read / time)
                bytes_read = size * 4  # 4 bytes per float32
                elapsed_time = end_time - start_time
                bandwidth_gb_s = (bytes_read / elapsed_time) / 1e9
                
                results['memory_bandwidth_gb_s'] = bandwidth_gb_s
                
                del data
            except:
                pass
                
    except Exception as e:
        results['error'] = str(e)
    
    return results

def get_memory_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get memory information for device.
    
    Args:
        device: Device to query (defaults to optimal device)
        
    Returns:
        Dictionary with memory information
        
    Examples:
        >>> mem_info = get_memory_info()
        >>> print(f"Available: {mem_info['available_gb']:.1f}GB")
    """
    if device is None:
        device = detect_optimal_device()
    
    info = {
        'device': str(device),
        'total_gb': 0.0,
        'available_gb': 0.0,
        'used_gb': 0.0,
        'cached_gb': 0.0
    }
    
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            cached = torch.cuda.memory_reserved(device)
            
            info.update({
                'total_gb': total / 1e9,
                'used_gb': allocated / 1e9,
                'cached_gb': cached / 1e9,
                'available_gb': (total - allocated) / 1e9
            })
            
        except Exception as e:
            info['error'] = str(e)
            
    elif device.type == 'mps':
        info.update({
            'total_gb': 'Unified',
            'available_gb': 'Unified',
            'used_gb': 'N/A',
            'cached_gb': 'N/A'
        })
    
    return info