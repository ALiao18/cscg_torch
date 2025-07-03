# V100 Optimization Plan for 200k+ Sequences

## Current Performance Issues
- 4.5h (CPU) → 6h (GPU) = 33% SLOWER on GPU
- Major bottlenecks identified in forward/backward passes and updateC

## Critical Optimizations for V100

### 1. Remove Debug Overhead (HIGH IMPACT) ✅ IMPLEMENTED
**Problem**: 15+ assertions per timestep × 200k = 3M assertion checks
**Solution**: 
```python
# Control via environment variable CHMM_DEBUG=1
DEBUG_MODE = os.environ.get('CHMM_DEBUG', '0') == '1'
if DEBUG_MODE:
    assert conditions...
```
**Usage**: 
- Production: Default (fast)
- Debug: `CHMM_DEBUG=1 python your_script.py`
**Expected gain**: 20-30% speed improvement

### 2. Vectorize Index Computations (HIGH IMPACT) ✅ IMPLEMENTED
**Problem**: `state_loc[i:i+2]` computed for every timestep
**Solution**: Pre-compute all indices at once
```python
# Pre-compute all indices vectorized
i_indices = x[:-1]  # [T-1]
j_indices = x[1:]   # [T-1] 
a_indices = a[:-1]  # [T-1]
i_starts = state_loc[i_indices]
i_stops = state_loc[i_indices + 1]
```
**Expected gain**: 15-25% speed improvement

### 3. Batch Matrix Operations (HIGH IMPACT) ✅ IMPLEMENTED
**Problem**: Sequential `torch.mv` calls and inefficient `torch.mul(torch.mul())` 
**Solution**: Replaced with optimized operations
```python
# Old: torch.mul(torch.mul(alpha, T_slice), beta)
# New: torch.outer(alpha, beta) * T_slice  # Better V100 utilization
```
**Expected gain**: 10-20% speed improvement

### 4. Memory Layout Optimization (HIGH IMPACT) ✅ IMPLEMENTED
**Problem**: Frequent tensor allocation creates memory fragmentation
**Solution**: Workspace tensor reuse system
```python
# Reuse pre-allocated tensors via workspace
if 'state_loc' not in workspace:
    workspace['state_loc'] = torch.cat([...]).cumsum(0)
state_loc = workspace['state_loc']
```

### 5. Mixed Precision Training (LOW IMPACT for V100)
**Problem**: V100 has Tensor Cores optimized for FP16
**Solution**: Use autocast more aggressively
```python
with torch.cuda.amp.autocast():
    # Core computations here
```
**Expected gain**: 5-15% speed improvement

## Implementation Priority

1. **Phase 1**: Remove assertions (immediate 20-30% gain)
2. **Phase 2**: Vectorize indices (15-25% additional gain)  
3. **Phase 3**: Batch operations (10-20% additional gain)
4. **Phase 4**: Memory optimization (5-10% additional gain)

## Target Performance
- **Goal**: 4.5h → 2-3h (50-33% of CPU time)
- **Realistic**: Achieve 3-4x speedup over current GPU implementation
- **Stretch**: Match or beat CPU performance significantly

## V100-Specific Features to Leverage
- 5,120 CUDA cores for parallel processing
- 32GB HBM2 memory (640 GB/s bandwidth)  
- Tensor Cores for mixed precision
- NVLink for multi-GPU scaling (if available)

## Validation Plan
Test on Colab V100 with:
- 16 obs, 20x20 room, 150 clones per obs
- Sequence lengths: 10k, 50k, 150k
- Compare against baseline CPU performance