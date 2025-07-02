# 📚 CSCG-Torch Examples

This directory contains examples and scripts to help you get started with CSCG-Torch.

## 🚀 Quick Start

### Google Colab (Recommended)
Open the interactive Colab notebook for the best experience:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/cscg-torch/blob/main/examples/CSCG_Torch_Colab_Demo.ipynb)

### Local Quick Start
```bash
cd examples
python quick_start_example.py
```

## 📁 Directory Structure

```
examples/
├── README.md                          # This file
├── CSCG_Torch_Colab_Demo.ipynb       # Interactive Colab notebook
├── quick_start_example.py             # Simple getting started script
└── scripts/                           # Advanced scripts
    ├── train_room.py                  # Full featured training script
    ├── test_v100_optimization.py      # V100 GPU optimization tests
    ├── test_long_gpu.py               # Long sequence generation tests
    ├── test_simple_gpu.py             # Basic GPU functionality tests
    └── test_small_training.py         # Quick training validation
```

## 📖 Example Descriptions

### 🎯 For Beginners

**`quick_start_example.py`** - Perfect first example
- Auto-detects your GPU and optimizes settings
- Loads room data and creates environment
- Demonstrates GPU-accelerated sequence generation
- Trains a small CHMM model
- Shows performance metrics
- ~5 minutes to run

**`CSCG_Torch_Colab_Demo.ipynb`** - Interactive tutorial
- Step-by-step guided experience
- GPU benchmarking and optimization
- Visualizations and plots
- Perfect for learning and experimentation
- Works in Google Colab with free GPU

### ⚡ For Power Users

**`scripts/train_room.py`** - Production training script
```bash
# Quick test
python scripts/train_room.py --seq_len 10000 --n_clones_per_obs 50 --n_iter 25

# Large scale training
python scripts/train_room.py --seq_len 500000 --n_clones_per_obs 200 --n_iter 100 --learn_E --verbose

# V100 optimized training
python scripts/train_room.py --seq_len 1000000 --n_clones_per_obs 500 --enable_mixed_precision
```

**`scripts/test_v100_optimization.py`** - GPU optimization diagnostics
- Benchmark your specific GPU
- Test V100/A100 specific optimizations
- Memory usage analysis
- Performance profiling

**`scripts/test_long_gpu.py`** - Stress test GPU sequence generation
- Tests sequences up to 1M+ steps
- Performance scaling analysis
- Memory efficiency validation

## 🛠️ Usage Patterns

### Basic Pattern
```python
import cscg_torch

# Auto-setup
device = cscg_torch.detect_optimal_device()
room_data = cscg_torch.load_room_data("room_20x20")
adapter = cscg_torch.create_room_adapter(room_data)

# Generate and train
x_seq, a_seq = adapter.generate_sequence_gpu(50000)
model, progression = cscg_torch.train_chmm(n_clones, x_seq, a_seq)
```

### Advanced Pattern
```python
import cscg_torch

# Detailed optimization
device = cscg_torch.detect_optimal_device()
gpu_settings = cscg_torch.optimize_for_gpu(device)
gpu_info = cscg_torch.get_gpu_info(device)

# Custom room and large scale
room_data = cscg_torch.create_random_room(50, 50, seed=42)
adapter = cscg_torch.create_room_adapter(room_data, adapter_type="torch")

# Optimized generation
x_seq, a_seq = adapter.generate_sequence_gpu(
    1_000_000, 
    device=device
)

# Advanced training
model, progression = cscg_torch.train_chmm(
    n_clones=cscg_torch.get_room_n_clones(n_clones_per_obs=500),
    x=x_seq,
    a=a_seq,
    device=device,
    method='em_T',
    n_iter=100,
    enable_mixed_precision=gpu_settings['mixed_precision'],
    learn_E=True,
    early_stopping=True
)

# Visualization
fig = cscg_torch.plot_training_progression(progression)
fig.show()
```

## 🎯 Recommended Learning Path

1. **Start Here**: `quick_start_example.py`
   - Run locally to verify installation
   - Understand basic API and workflow

2. **Interactive Learning**: `CSCG_Torch_Colab_Demo.ipynb`
   - Open in Google Colab
   - Follow step-by-step tutorial
   - Experiment with parameters

3. **Production Usage**: `scripts/train_room.py`
   - Use command-line interface
   - Scale up to larger problems
   - Optimize for your specific GPU

4. **Performance Tuning**: `scripts/test_v100_optimization.py`
   - Benchmark your hardware
   - Understand optimization settings
   - Maximize performance

## 🔧 GPU Optimization Examples

### Automatic Optimization
```python
# Let CSCG-Torch optimize automatically
device = cscg_torch.detect_optimal_device()
settings = cscg_torch.optimize_for_gpu(device)

# Use optimized settings
x_seq, a_seq = adapter.generate_sequence_gpu(
    length=settings['chunk_size'] * 10,  # Scale with chunk size
    device=device
)
```

### Manual V100 Optimization
```python
# Manual optimization for V100
if "V100" in gpu_info['name']:
    chunk_size = 65_536  # Large chunks for V100
    mixed_precision = True
    batch_multiplier = 2.0
else:
    chunk_size = 16_384
    mixed_precision = False
    batch_multiplier = 1.0

# Apply settings
model, progression = cscg_torch.train_chmm(
    n_clones, x_seq, a_seq,
    enable_mixed_precision=mixed_precision,
    device=device
)
```

## 📊 Performance Expectations

### Sequence Generation (GPU-accelerated)
- **V100**: 400-600K steps/second
- **A100**: 600-1M steps/second  
- **RTX 4090**: 300-500K steps/second
- **T4 (Colab)**: 100-200K steps/second
- **MPS (Apple)**: 200-400K steps/second

### CHMM Training
- **Small (50K steps, 1K states)**: 10-30 seconds
- **Medium (500K steps, 5K states)**: 2-5 minutes
- **Large (1M+ steps, 10K+ states)**: 10-30 minutes

## 🐛 Troubleshooting

### Common Issues

**ImportError**: Make sure CSCG-Torch is properly installed
```bash
pip install -e .  # From the cscg_torch/ directory
```

**CUDA Out of Memory**: Reduce sequence length or number of clones
```python
# Reduce parameters
seq_len = 25_000  # Instead of 100_000
n_clones_per_obs = 50  # Instead of 200
```

**Slow Performance**: Check GPU optimization
```python
gpu_info = cscg_torch.get_gpu_info()
print("Optimizations:", gpu_info['optimizations'])
```

### Getting Help

1. Check the [main README](../README.md)
2. Run the diagnostic scripts in `scripts/`
3. Open an issue on GitHub with your error output

## 🎉 Next Steps

After running the examples:

- Experiment with different room sizes and parameters
- Try your own custom environments
- Scale up to larger sequences and models
- Integrate CSCG-Torch into your research projects

Happy researching! 🚀