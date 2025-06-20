# CSCG Refactor Plan

## Objective

Refactor the Clone-Structured Cognitive Graph (CSCG) model into a modular, GPU-accelerated PyTorch implementation that supports batched inference, scalable EM/Viterbi training, and integration into reinforcement learning frameworks.

As you refactor and make any changes, verbosely log your changes into refactor.md. 

---

## Modular Structure (Proposed)

| Module                      | Purpose                                                   |
|----------------------------|-----------------------------------------------------------|
| `models/chmm.py`           | Core CHMM class: init, sample, decode, bridge             |
| `models/chmm_utils.py`     | Utility functions: clone indexing, input validation       |
| `inference/forward.py`     | Forward, backward, max-product (Viterbi) message passing  |
| `training/em.py`           | EM and Viterbi training loops                             |
| `data/generator.py`        | Maze generation and random walk simulation                |
| `utils/visualization.py`   | Plotting: graphs, place fields, clone usage, BPS curves   |
| `tests/`                   | Unit and integration tests                                |

---

## Tensor Conventions (PyTorch)

| Symbol | Shape          | Description                          |
|--------|----------------|--------------------------------------|
| `T`    | `[A, S, S]`    | Transition tensor (action-conditioned) |
| `E`    | `[S, O]`       | Emission matrix                     |
| `x`    | `[T]`          | Observation sequence (ints)         |
| `a`    | `[T]`          | Action sequence (ints)              |
| `n_clones` | `[O]`      | Clone count per observation         |
| `mess_fwd`, `mess_bwd` | `[T, S]` | Forward / backward messages   |

- All tensors default to: `dtype=torch.float32`, `device='cuda'`
- Use `.to(device)` and `.detach().cpu()` explicitly at interfaces

---

## Defensive Programming Standards

Each core function **must**:

- Accept `verbose: bool = False`
- Print:
  - Shape
  - Dtype
  - Device
- Validate:
  - Shape/dtype compatibility
  - Device consistency
  - Sequence length agreement
- Example:

```python
assert x.shape == a.shape, f"x and a must be same shape: {x.shape}, {a.shape}"
assert x.dtype == torch.long
assert x.device == device
