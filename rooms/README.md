# CSCG Test Rooms

This directory contains test room layouts for CSCG environment adapters.

## Room Specifications

- **Number of observation states**: 16 (0-15)
- **Wall representation**: -1
- **Data types**: NumPy arrays (.npy) and PyTorch tensors (.pt)

## Available Rooms

### 5x5 Room
- **Files**: `room_5x5_16states.{npy,pt,txt}`
- **Dimensions**: 5 x 5
- **Total cells**: 25
- **Use case**: Fast testing

### 10x10 Room
- **Files**: `room_10x10_16states.{npy,pt,txt}`
- **Dimensions**: 10 x 10
- **Total cells**: 100
- **Use case**: Standard testing

### 20x20 Room
- **Files**: `room_20x20_16states.{npy,pt,txt}`
- **Dimensions**: 20 x 20
- **Total cells**: 400
- **Use case**: Standard testing

### 50x50 Room
- **Files**: `room_50x50_16states.{npy,pt,txt}`
- **Dimensions**: 50 x 50
- **Total cells**: 2500
- **Use case**: Performance testing

## Usage Examples

### Loading with NumPy
```python
import numpy as np
room = np.load('room_5x5_16states.npy')
```

### Loading with PyTorch
```python
import torch
room = torch.load('room_5x5_16states.pt')
```

### Using with CSCG Adapters
```python
from cscg_torch.env_adapters.room_utils import create_room_adapter
import torch

# Load room data
room = torch.load('room_5x5_16states.pt')

# Create adapter
adapter = create_room_adapter(room, adapter_type='torch')

# Generate test sequences
x_seq, a_seq = adapter.generate_sequence(2000)
```

## Compatibility

- Compatible with `RoomNPAdapter` and `RoomTorchAdapter`
- Works with `create_room_adapter()` utility function
- Suitable for testing CHMM training and inference
