# CSCG PyTorch

A GPU-optimized PyTorch implementation of Compositional Structured Context-Sensitive Grammar (CSCG) Hidden Markov Models using the Baum-Welch EM algorithm.

## Features

- **GPU-Accelerated**: Full CUDA support with optimized tensor operations
- **EM Training**: Efficient Baum-Welch algorithm implementation
- **Viterbi Decoding**: MAP inference with max-product message passing
- **Flexible Architecture**: Support for emissions and action-dependent transitions
- **Environment Adapters**: Easy integration with different domains

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.1.0+
- CUDA-compatible GPU (recommended)

### From Source

```bash
git clone <repository-url>
cd cscg_torch
pip install -r requirements.txt
```

### Google Colab Setup

```python
# Install dependencies
!pip install torch>=2.1.0 numpy>=1.24 matplotlib>=3.7 tqdm>=4.65

# Clone and setup
!git clone <repository-url>
import sys
sys.path.append('/content/cscg_torch')
```

## Quick Start

### Basic Usage

```python
import torch
import numpy as np
from cscg_torch import CHMM_torch

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_clones = torch.tensor([2, 3, 2], dtype=torch.int64)  # clones per observation
x = torch.tensor([0, 1, 2, 1, 0], dtype=torch.int64)  # observations
a = torch.tensor([0, 1, 0, 1], dtype=torch.int64)     # actions

# Create model
model = CHMM_torch(n_clones, x, a, dtype=torch.float32)

# Train with EM
convergence = model.learn_em_T(x, a, n_iter=50)

# Compute likelihood
bps = model.bps(x, a)
print(f"Bits per step: {bps:.3f}")

# Decode MAP sequence
log_lik, states = model.decode(x, a)
print(f"MAP states: {states}")
```

### Training with Emissions

```python
# Learn emission matrix
convergence, E = model.learn_em_E(x, a, n_iter=50)

# Use learned emissions for inference
bps_e = model.bpsE(E, x, a)
log_lik_e, states_e = model.decodeE(E, x, a)
```

### Sequence Generation

```python
# Sample from learned model
sample_x, sample_a = model.sample(length=100)

# Conditional sampling
sequence = model.sample_sym(start_symbol=0, length=50)
```

## API Reference

### CHMM_torch Class

**Constructor**
```python
CHMM_torch(n_clones, x, a, pseudocount=0.0, dtype=torch.float32, seed=42)
```

**Training Methods**
- `learn_em_T(x, a, n_iter=100)`: EM training for transitions
- `learn_em_E(x, a, n_iter=100)`: EM training for emissions  
- `learn_viterbi_T(x, a, n_iter=100)`: Hard EM (Viterbi) training

**Inference Methods**
- `bps(x, a)`: Compute bits-per-step likelihood
- `bpsE(E, x, a)`: Likelihood with custom emissions
- `decode(x, a)`: MAP decoding
- `decodeE(E, x, a)`: MAP decoding with emissions

**Generation Methods**
- `sample(length)`: Generate sequences
- `sample_sym(sym, length)`: Conditional generation
- `bridge(state1, state2)`: Path between states

## Environment Adapters

```python
from cscg_torch.env_adapters import RoomAdapter

# Create environment adapter
env = RoomAdapter(grid_size=5, n_rooms=4)

# Generate training data
x_seq, a_seq = env.generate_sequence(length=1000)

# Train model
model = CHMM_torch(env.n_clones, x_seq, a_seq)
convergence = model.learn_em_T(x_seq, a_seq, n_iter=100)
```

## GPU Optimization Features

- **Vectorized Operations**: All loops replaced with tensor operations
- **Device Management**: Automatic GPU/CPU placement
- **Memory Efficient**: Optimized tensor allocation and reuse
- **Batched Processing**: Support for sequence batches (future release)

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for best performance
2. **Appropriate Data Types**: Use `torch.float32` for speed, `torch.float64` for precision
3. **Sequence Length**: Longer sequences benefit more from GPU acceleration
4. **Memory Management**: Monitor GPU memory usage for large models

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce model size or use CPU
model = CHMM_torch(n_clones, x, a, device=torch.device("cpu"))
```

**Numerical Instability**
```python
# Increase pseudocount
model = CHMM_torch(n_clones, x, a, pseudocount=1e-10)
```

**Google Colab GPU Access**
```python
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Examples

See the `examples/` directory for:
- Basic training workflow
- Advanced emission learning
- Environment integration
- Performance benchmarks

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cscg_torch,
  title={CSCG PyTorch: GPU-Optimized Implementation},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/cscg_torch}
}
```

## License

See LICENSE file for details.