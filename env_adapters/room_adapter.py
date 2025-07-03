# env_adapters/room_adapter.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from .base_adapter import CSCGEnvironmentAdapter

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

class RoomNPAdapter(CSCGEnvironmentAdapter):
    def __init__(self, room_array, no_up=[], no_down=[], no_left=[], no_right=[], start_pos=None, seed=42):
        # Strict input validation
        assert isinstance(room_array, np.ndarray), f"room_array must be numpy array, got {type(room_array)}"
        assert room_array.ndim == 2, f"room_array must be 2D, got {room_array.ndim}D"
        assert room_array.dtype in [np.int32, np.int64], f"room_array must be integer type, got {room_array.dtype}"
        assert room_array.size > 0, "room_array cannot be empty"
        
        super().__init__(seed=seed)
        
        self.room = room_array
        self.h, self.w = int(self.room.shape[0]), int(self.room.shape[1])
        
        # Validate dimensions
        assert self.h > 0 and self.w > 0, f"Invalid room dimensions: {self.h}x{self.w}"
        
        # Validate start position
        if start_pos is not None:
            assert isinstance(start_pos, (tuple, list)), f"start_pos must be tuple/list, got {type(start_pos)}"
            assert len(start_pos) == 2, f"start_pos must have 2 elements, got {len(start_pos)}"
            r, c = start_pos
            assert isinstance(r, int) and isinstance(c, int), f"start_pos elements must be int, got {type(r)}, {type(c)}"
            assert 0 <= r < self.h and 0 <= c < self.w, f"start_pos {start_pos} out of bounds for {self.h}x{self.w} room"
        
        self.start_pos = start_pos
        
        # Validate wall lists
        for wall_list, name in [(no_up, "no_up"), (no_down, "no_down"), (no_left, "no_left"), (no_right, "no_right")]:
            assert isinstance(wall_list, (list, tuple, set)), f"{name} must be list/tuple/set, got {type(wall_list)}"
            for idx in wall_list:
                assert isinstance(idx, int), f"{name} elements must be int, got {type(idx)}"
                assert 0 <= idx < self.h * self.w, f"{name} index {idx} out of bounds for {self.h}x{self.w} room"
        
        self.no_up = set(no_up)
        self.no_down = set(no_down)
        self.no_left = set(no_left)
        self.no_right = set(no_right)
        # Support CUDA, MPS (Apple Silicon), and CPU with consistent device naming
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps:0")
        else:
            self.device = torch.device("cpu")

        self.action_map = ACTIONS
        self.n_actions = int(4)
        
        # Post-initialization validation
        assert isinstance(self.n_actions, int), f"n_actions must be int, got {type(self.n_actions)}"
        assert self.n_actions == len(ACTIONS), f"n_actions mismatch: {self.n_actions} != {len(ACTIONS)}"
        
        self.reset()

    def reset(self):
        # Strict validation for reset operation
        assert hasattr(self, 'room'), "room must be initialized"
        assert hasattr(self, 'rng'), "rng must be initialized"
        assert isinstance(self.room, np.ndarray), f"room must be numpy array, got {type(self.room)}"
        
        if self.start_pos is None:
            free_positions = np.argwhere(self.room != -1)
            assert len(free_positions) > 0, "No free positions found in room"
            assert isinstance(free_positions, np.ndarray), f"free_positions must be numpy array, got {type(free_positions)}"
            
            idx = self.rng.choice(len(free_positions))
            assert isinstance(idx, (int, np.integer)), f"idx must be int, got {type(idx)}"
            assert 0 <= idx < len(free_positions), f"idx {idx} out of range [0, {len(free_positions)})"
            
            selected_pos = free_positions[idx]
            assert len(selected_pos) == 2, f"selected_pos must have 2 elements, got {len(selected_pos)}"
            self.pos = tuple(int(x) for x in selected_pos)
        else:
            assert self.start_pos is not None, "start_pos validation failed"
            self.pos = tuple(self.start_pos)
        
        # Validate final position
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        assert 0 <= r < self.h and 0 <= c < self.w, f"pos {self.pos} out of bounds for {self.h}x{self.w} room"
        
        return self.get_observation()

    def step(self, action):
        # Strict input validation
        assert isinstance(action, (int, np.integer)), f"action must be int, got {type(action)}"
        assert 0 <= action < self.n_actions, f"action {action} out of range [0, {self.n_actions})"
        assert action in self.action_map, f"action {action} not in action_map {list(self.action_map.keys())}"
        
        # Validate current state
        assert hasattr(self, 'pos'), "pos must be set (call reset() first)"
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        
        dr, dc = self.action_map[action]
        assert isinstance(dr, int) and isinstance(dc, int), f"action_map values must be int, got {type(dr)}, {type(dc)}"
        
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        
        new_r, new_c = r + dr, c + dc

        # Boundary check
        if not (0 <= new_r < self.h and 0 <= new_c < self.w):
            return self.get_observation(), False
        
        # Wall check
        if self.room[new_r, new_c] == -1:
            return self.get_observation(), False

        # Invisible wall checks
        flat_idx = r * self.w + c
        assert isinstance(flat_idx, int), f"flat_idx must be int, got {type(flat_idx)}"
        assert 0 <= flat_idx < self.h * self.w, f"flat_idx {flat_idx} out of bounds for {self.h}x{self.w} room"
        
        if action == 0 and flat_idx in self.no_up:
            return self.get_observation(), False
        if action == 1 and flat_idx in self.no_down:
            return self.get_observation(), False
        if action == 2 and flat_idx in self.no_left:
            return self.get_observation(), False
        if action == 3 and flat_idx in self.no_right:
            return self.get_observation(), False

        # Update position
        self.pos = (int(new_r), int(new_c))
        return self.get_observation(), True

    def get_observation(self):
        # Strict validation for observation generation
        assert hasattr(self, 'pos'), "pos must be set (call reset() first)"
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        assert 0 <= r < self.h and 0 <= c < self.w, f"pos {self.pos} out of bounds for {self.h}x{self.w} room"
        
        # Check walls in all directions
        up    = int(self.room[r - 1, c] != -1) if r > 0     else 0
        down  = int(self.room[r + 1, c] != -1) if r < self.h - 1 else 0
        left  = int(self.room[r, c - 1] != -1) if c > 0     else 0
        right = int(self.room[r, c + 1] != -1) if c < self.w - 1 else 0
        
        # Validate wall check results
        for val, name in [(up, "up"), (down, "down"), (left, "left"), (right, "right")]:
            assert isinstance(val, int), f"{name} must be int, got {type(val)}"
            assert val in [0, 1], f"{name} must be 0 or 1, got {val}"
        
        obs = (up << 3) + (down << 2) + (left << 1) + right
        
        # Final validation
        assert isinstance(obs, int), f"obs must be int, got {type(obs)}"
        assert 0 <= obs <= 15, f"obs {obs} out of range [0, 15]"
        
        return obs

    def generate_sequence_gpu(self, length, device=None):
        """
        GPU-accelerated sequence generation for numpy room adapter.
        
        Args:
            length (int): Sequence length to generate
            device (torch.device, optional): Device for computation
            
        Returns:
            tuple: (x_seq, a_seq) as numpy arrays
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                device = torch.device("mps:0")
            else:
                device = torch.device("cpu")
        
        # Convert room to tensor for GPU computation
        room_tensor = torch.tensor(self.room, device=device, dtype=torch.int64)
        
        # Create temporary torch adapter for GPU computation
        temp_adapter = RoomTorchAdapter(
            room_tensor, 
            no_up=list(self.no_up),
            no_down=list(self.no_down), 
            no_left=list(self.no_left),
            no_right=list(self.no_right),
            start_pos=self.start_pos,
            seed=self.rng.get_state()[1][0]  # Extract seed from numpy state
        )
        
        # Use torch adapter's GPU method
        return temp_adapter.generate_sequence_gpu(length, device)

class RoomTorchAdapter(CSCGEnvironmentAdapter):
    def __init__(self, room_tensor, no_up=[], no_down=[], no_left=[], no_right=[], start_pos=None, seed=42):
        # Strict input validation for PyTorch tensor
        assert isinstance(room_tensor, torch.Tensor), f"room_tensor must be torch.Tensor, got {type(room_tensor)}"
        assert room_tensor.ndim == 2, f"room_tensor must be 2D, got {room_tensor.ndim}D"
        assert room_tensor.dtype in [torch.int32, torch.int64, torch.long], f"room_tensor must be integer type, got {room_tensor.dtype}"
        assert room_tensor.numel() > 0, "room_tensor cannot be empty"
        
        super().__init__(seed=seed)
        
        # Ensure room tensor is on the correct device with consistent naming
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps:0")
        else:
            self.device = torch.device("cpu")
        self.room = room_tensor.to(self.device)
        self.h, self.w = int(self.room.shape[0]), int(self.room.shape[1])
        
        # Validate dimensions
        assert self.h > 0 and self.w > 0, f"Invalid room dimensions: {self.h}x{self.w}"
        
        # Validate start position
        if start_pos is not None:
            assert isinstance(start_pos, (tuple, list)), f"start_pos must be tuple/list, got {type(start_pos)}"
            assert len(start_pos) == 2, f"start_pos must have 2 elements, got {len(start_pos)}"
            r, c = start_pos
            assert isinstance(r, int) and isinstance(c, int), f"start_pos elements must be int, got {type(r)}, {type(c)}"
            assert 0 <= r < self.h and 0 <= c < self.w, f"start_pos {start_pos} out of bounds for {self.h}x{self.w} room"
        
        self.start_pos = start_pos
        
        # Validate wall lists
        for wall_list, name in [(no_up, "no_up"), (no_down, "no_down"), (no_left, "no_left"), (no_right, "no_right")]:
            assert isinstance(wall_list, (list, tuple, set)), f"{name} must be list/tuple/set, got {type(wall_list)}"
            for idx in wall_list:
                assert isinstance(idx, int), f"{name} elements must be int, got {type(idx)}"
                assert 0 <= idx < self.h * self.w, f"{name} index {idx} out of bounds for {self.h}x{self.w} room"
        
        self.no_up = set(no_up)
        self.no_down = set(no_down)
        self.no_left = set(no_left)
        self.no_right = set(no_right)

        self.action_map = ACTIONS
        self.n_actions = 4
        
        # Post-initialization validation
        assert isinstance(self.n_actions, int), f"n_actions must be int, got {type(self.n_actions)}"
        assert self.n_actions == len(ACTIONS), f"n_actions mismatch: {self.n_actions} != {len(ACTIONS)}"
        assert self.room.device.type == self.device.type, f"Room tensor device type mismatch: {self.room.device} != {self.device}"
        
        self.reset()

    def reset(self):
        # Strict validation for reset operation
        assert hasattr(self, 'room'), "room must be initialized"
        assert hasattr(self, 'rng'), "rng must be initialized"
        assert isinstance(self.room, torch.Tensor), f"room must be torch.Tensor, got {type(self.room)}"
        assert self.room.device.type == self.device.type, f"room device type mismatch: {self.room.device} != {self.device}"
        
        if self.start_pos is None:
            free_positions = (self.room != -1).nonzero(as_tuple=False)
            assert isinstance(free_positions, torch.Tensor), f"free_positions must be torch.Tensor, got {type(free_positions)}"
            assert len(free_positions) > 0, "No free positions found in room"
            
            idx = self.rng.choice(len(free_positions))
            assert isinstance(idx, (int, np.integer)), f"idx must be int, got {type(idx)}"
            assert 0 <= idx < len(free_positions), f"idx {idx} out of range [0, {len(free_positions)})"
            
            selected_pos = free_positions[idx].tolist()
            assert isinstance(selected_pos, list), f"selected_pos must be list, got {type(selected_pos)}"
            assert len(selected_pos) == 2, f"selected_pos must have 2 elements, got {len(selected_pos)}"
            self.pos = tuple(int(x) for x in selected_pos)
        else:
            assert self.start_pos is not None, "start_pos validation failed"
            self.pos = tuple(self.start_pos)
        
        # Validate final position
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        assert 0 <= r < self.h and 0 <= c < self.w, f"pos {self.pos} out of bounds for {self.h}x{self.w} room"
        
        return self.get_observation()

    def step(self, action):
        # Strict input validation
        assert isinstance(action, (int, np.integer)), f"action must be int, got {type(action)}"
        assert 0 <= action < self.n_actions, f"action {action} out of range [0, {self.n_actions})"
        assert action in self.action_map, f"action {action} not in action_map {list(self.action_map.keys())}"
        
        # Validate current state
        assert hasattr(self, 'pos'), "pos must be set (call reset() first)"
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        assert hasattr(self, 'room'), "room must be initialized"
        assert self.room.device.type == self.device.type, f"room device type mismatch: {self.room.device} != {self.device}"
        
        dr, dc = self.action_map[action]
        assert isinstance(dr, int) and isinstance(dc, int), f"action_map values must be int, got {type(dr)}, {type(dc)}"
        
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        
        new_r, new_c = r + dr, c + dc

        # Boundary check
        if not (0 <= new_r < self.h and 0 <= new_c < self.w):
            return self.get_observation(), False
        
        # Wall check (tensor comparison)
        wall_check = self.room[new_r, new_c] == -1
        assert isinstance(wall_check, torch.Tensor), f"wall_check must be tensor, got {type(wall_check)}"
        if wall_check.item():
            return self.get_observation(), False

        # Invisible wall checks
        flat_idx = r * self.w + c
        assert isinstance(flat_idx, int), f"flat_idx must be int, got {type(flat_idx)}"
        assert 0 <= flat_idx < self.h * self.w, f"flat_idx {flat_idx} out of bounds for {self.h}x{self.w} room"
        
        if action == 0 and flat_idx in self.no_up:
            return self.get_observation(), False
        if action == 1 and flat_idx in self.no_down:
            return self.get_observation(), False
        if action == 2 and flat_idx in self.no_left:
            return self.get_observation(), False
        if action == 3 and flat_idx in self.no_right:
            return self.get_observation(), False

        # Update position
        self.pos = (int(new_r), int(new_c))
        return self.get_observation(), True

    def get_observation(self):
        # Strict validation for observation generation
        assert hasattr(self, 'pos'), "pos must be set (call reset() first)"
        assert isinstance(self.pos, tuple), f"pos must be tuple, got {type(self.pos)}"
        assert len(self.pos) == 2, f"pos must have 2 elements, got {len(self.pos)}"
        assert hasattr(self, 'room'), "room must be initialized"
        assert self.room.device.type == self.device.type, f"room device type mismatch: {self.room.device} != {self.device}"
        
        r, c = self.pos
        assert isinstance(r, int) and isinstance(c, int), f"pos elements must be int, got {type(r)}, {type(c)}"
        assert 0 <= r < self.h and 0 <= c < self.w, f"pos {self.pos} out of bounds for {self.h}x{self.w} room"
        
        # Check walls in all directions (convert tensor comparisons to Python ints)
        up    = int((self.room[r - 1, c] != -1).item()) if r > 0     else 0
        down  = int((self.room[r + 1, c] != -1).item()) if r < self.h - 1 else 0
        left  = int((self.room[r, c - 1] != -1).item()) if c > 0     else 0
        right = int((self.room[r, c + 1] != -1).item()) if c < self.w - 1 else 0
        
        # Validate wall check results
        for val, name in [(up, "up"), (down, "down"), (left, "left"), (right, "right")]:
            assert isinstance(val, int), f"{name} must be int, got {type(val)}"
            assert val in [0, 1], f"{name} must be 0 or 1, got {val}"
        
        obs = (up << 3) + (down << 2) + (left << 1) + right
        
        # Final validation
        assert isinstance(obs, int), f"obs must be int, got {type(obs)}"
        assert 0 <= obs <= 15, f"obs {obs} out of range [0, 15]"
        
        return obs

    def generate_sequence_gpu(self, length: int, device: torch.device = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate navigation sequence using GPU-accelerated vectorized operations.
        
        This method pre-computes lookup tables for observations and transitions,
        then processes the sequence in chunks to minimize CPU-GPU synchronization.
        Optimized for both MPS (Apple Silicon) and CUDA (V100/A100) devices.
        
        Args:
            length: Number of time steps to generate
            device: Target device for computation (auto-detected if None)
            
        Returns:
            Tuple of (observations, actions) as numpy arrays of shape (length,)
            
        Raises:
            ValueError: If no valid starting positions exist in the room
            RuntimeError: If GPU computation fails
        """
        # Device setup and validation
        device = self._setup_device(device)
        self._validate_sequence_parameters(length)
        
        # GPU computation pipeline
        lookup_tables = self._precompute_lookup_tables(device)
        sequence_tensors = self._allocate_sequence_tensors(length, device)
        starting_position = self._initialize_starting_position(device)
        
        # Generate sequence using vectorized processing
        self._generate_sequence_vectorized(
            length=length,
            starting_position=starting_position,
            lookup_tables=lookup_tables,
            sequence_tensors=sequence_tensors,
            device=device
        )
        
        # Return results as numpy arrays
        return self._finalize_sequence_output(sequence_tensors)
    
    def _setup_device(self, device: torch.device = None) -> torch.device:
        """Setup and validate compute device."""
        if device is None:
            device = self.device
        
        # Log device information for research reproducibility
        device_info = {
            'type': device.type,
            'index': getattr(device, 'index', None),
            'name': device
        }
        print(f"Using device: {device_info}")
        
        return device
    
    def _validate_sequence_parameters(self, length: int) -> None:
        """Validate sequence generation parameters."""
        if not isinstance(length, int):
            raise TypeError(f"Sequence length must be int, got {type(length)}")
        if length <= 0:
            raise ValueError(f"Sequence length must be positive, got {length}")
        if length > 10_000_000:  # Safety limit for memory
            raise ValueError(f"Sequence length {length} exceeds safety limit of 10M steps")
    
    def _precompute_lookup_tables(self, device: torch.device) -> dict:
        """
        Pre-compute all lookup tables for efficient GPU computation.
        
        Returns:
            Dictionary containing:
            - obs_table: [H, W] -> observation_id mapping
            - valid_actions_table: [H, W, 4] -> action validity mask  
            - transition_table: [H, W, 4, 2] -> next position mapping
        """
        print("Pre-computing GPU lookup tables...")
        
        observation_table = self._build_observation_lookup_table(device)
        action_validity_table, position_transition_table = self._build_transition_lookup_tables(device)
        
        return {
            'obs_table': observation_table,
            'valid_actions_table': action_validity_table,
            'transition_table': position_transition_table
        }
    
    def _allocate_sequence_tensors(self, length: int, device: torch.device) -> dict:
        """Pre-allocate all tensors needed for sequence generation."""
        return {
            'observations': torch.empty(length, dtype=torch.int64, device=device),
            'actions': torch.empty(length, dtype=torch.int64, device=device),
            'positions': torch.empty((length, 2), dtype=torch.int64, device=device),
            'random_actions': torch.randint(0, 4, (length,), device=device)
        }
    
    def _initialize_starting_position(self, device: torch.device) -> torch.Tensor:
        """Determine starting position for the navigation sequence."""
        if self.start_pos is None:
            # Find all valid (non-wall) positions
            valid_positions = (self.room != -1).nonzero(as_tuple=False)
            if len(valid_positions) == 0:
                raise ValueError("No valid starting positions found in room layout")
            
            # Randomly select starting position
            random_index = torch.randint(0, len(valid_positions), (1,), device=device)
            starting_position = valid_positions[random_index[0]]
        else:
            starting_position = torch.tensor(self.start_pos, dtype=torch.int64, device=device)
        
        return starting_position
    
    def _generate_sequence_vectorized(self, length: int, starting_position: torch.Tensor,
                                    lookup_tables: dict, sequence_tensors: dict, 
                                    device: torch.device) -> None:
        """
        Generate the navigation sequence using vectorized GPU operations.
        
        Processes sequence in chunks to balance memory usage and computational efficiency.
        Chunk size is optimized for different GPU architectures.
        """
        # Set initial position
        sequence_tensors['positions'][0] = starting_position
        
        # Determine optimal chunk size based on device capabilities
        chunk_size = self._determine_optimal_chunk_size(device, length)
        
        print(f"Processing {length:,} steps in chunks of {chunk_size:,}...")
        
        from tqdm import tqdm
        with tqdm(total=length, desc="Generating sequence", 
                 unit="steps", ncols=100, colour="green", 
                 leave=True, dynamic_ncols=False, ascii=False, 
                 position=0, disable=False, mininterval=0.1) as progress_bar:
            
            for chunk_start in range(0, length, chunk_size):
                chunk_end = min(chunk_start + chunk_size, length)
                
                self._process_sequence_chunk(
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    lookup_tables=lookup_tables,
                    sequence_tensors=sequence_tensors,
                    device=device
                )
                
                progress_bar.update(chunk_end - chunk_start)
    
    def _determine_optimal_chunk_size(self, device: torch.device, sequence_length: int) -> int:
        """Determine optimal chunk size based on device and sequence characteristics."""
        if device.type == 'cuda':
            return self._get_cuda_optimized_chunk_size(device, sequence_length)
        
        # Base chunk sizes for non-CUDA devices
        chunk_sizes = {
            'mps': 8_192,      # Apple Silicon - smaller chunks due to unified memory
            'cpu': 1_024       # CPU fallback
        }
        
        base_chunk_size = chunk_sizes.get(device.type, 1_024)
        
        # Adaptive sizing based on sequence length
        if sequence_length < 10_000:
            return min(base_chunk_size // 4, sequence_length)
        elif sequence_length > 1_000_000:
            return base_chunk_size * 2
        else:
            return base_chunk_size
    
    def _get_cuda_optimized_chunk_size(self, device: torch.device, sequence_length: int) -> int:
        """Get CUDA-optimized chunk size with V100-specific optimizations."""
        try:
            gpu_props = torch.cuda.get_device_properties(device)
            gpu_name = gpu_props.name
            total_memory_gb = gpu_props.total_memory / (1024**3)
            
            # V100-specific optimization (16GB HBM2, 900 GB/s bandwidth)
            if "V100" in gpu_name:
                if sequence_length > 1_000_000:
                    return 65_536  # Large chunks for massive sequences on V100
                elif sequence_length > 100_000:
                    return 32_768  # Medium-large chunks for V100's high bandwidth
                elif sequence_length > 50_000:
                    return 24_576  # Balanced chunks for V100
                else:
                    return 16_384  # Standard chunks for smaller sequences
            
            # A100/H100 optimization (40GB+ HBM, higher bandwidth)
            elif any(gpu in gpu_name for gpu in ["A100", "H100"]):
                if sequence_length > 1_000_000:
                    return 131_072  # Very large chunks for newer GPUs
                elif sequence_length > 100_000:
                    return 65_536
                else:
                    return 32_768
            
            # General CUDA optimization based on memory
            else:
                memory_factor = min(total_memory_gb / 16.0, 2.0)  # Relative to V100's 16GB
                base_chunk = 16_384
                
                if sequence_length > 1_000_000:
                    return int(base_chunk * 2 * memory_factor)
                elif sequence_length > 100_000:
                    return int(base_chunk * 1.5 * memory_factor)
                else:
                    return int(base_chunk * memory_factor)
                    
        except Exception:
            # Fallback if GPU properties unavailable
            if sequence_length > 1_000_000:
                return 32_768
            elif sequence_length > 100_000:
                return 24_576
            else:
                return 16_384
    
    def _finalize_sequence_output(self, sequence_tensors: dict) -> tuple[np.ndarray, np.ndarray]:
        """Convert GPU tensors to numpy arrays for return."""
        print("Converting results to CPU numpy arrays...")
        
        observations = sequence_tensors['observations'].cpu().numpy()
        actions = sequence_tensors['actions'].cpu().numpy()
        
        # Validate output integrity
        assert observations.dtype == np.int64, f"Unexpected observations dtype: {observations.dtype}"
        assert actions.dtype == np.int64, f"Unexpected actions dtype: {actions.dtype}"
        assert len(observations) == len(actions), "Sequence length mismatch"
        
        return observations, actions
    
    def _build_observation_lookup_table(self, device: torch.device) -> torch.Tensor:
        """
        Build lookup table mapping room positions to 4-bit observation codes.
        
        Each observation encodes wall availability in 4 directions:
        - Bit 3 (×8): up direction passable
        - Bit 2 (×4): down direction passable  
        - Bit 1 (×2): left direction passable
        - Bit 0 (×1): right direction passable
        
        Args:
            device: Target device for tensor computation
            
        Returns:
            Tensor of shape [H, W] with observation codes (0-15)
        """
        room_height, room_width = self.h, self.w
        
        # Transfer room layout to target device
        room_on_device = self.room.to(device)
        
        # Initialize direction passability tensors
        direction_passable = {
            'up': torch.zeros((room_height, room_width), dtype=torch.int64, device=device),
            'down': torch.zeros((room_height, room_width), dtype=torch.int64, device=device),
            'left': torch.zeros((room_height, room_width), dtype=torch.int64, device=device),
            'right': torch.zeros((room_height, room_width), dtype=torch.int64, device=device)
        }
        
        # Vectorized passability checking using tensor slicing
        # Up direction: check if cell above is passable (not wall = -1)
        direction_passable['up'][1:, :] = (room_on_device[:-1, :] != -1).long()
        
        # Down direction: check if cell below is passable
        direction_passable['down'][:-1, :] = (room_on_device[1:, :] != -1).long()
        
        # Left direction: check if cell to the left is passable
        direction_passable['left'][:, 1:] = (room_on_device[:, :-1] != -1).long()
        
        # Right direction: check if cell to the right is passable
        direction_passable['right'][:, :-1] = (room_on_device[:, 1:] != -1).long()
        
        # Encode as 4-bit observation codes (compatible with MPS and CUDA)
        observation_lookup_table = (
            direction_passable['up'] * 8 +      # Bit 3
            direction_passable['down'] * 4 +    # Bit 2  
            direction_passable['left'] * 2 +    # Bit 1
            direction_passable['right'] * 1     # Bit 0
        )
        
        return observation_lookup_table
    
    def _build_transition_lookup_tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build lookup tables for action validity and position transitions.
        
        Computes two complementary tables:
        1. Action validity: Which actions are legal from each position
        2. Position transitions: Where each action leads from each position
        
        Args:
            device: Target device for tensor computation
            
        Returns:
            Tuple of (action_validity_table, position_transition_table):
            - action_validity_table: [H, W, 4] boolean tensor indicating valid actions
            - position_transition_table: [H, W, 4, 2] tensor with resulting positions
        """
        room_height, room_width = self.h, self.w
        
        # Initialize lookup tables
        action_validity_table = torch.zeros((room_height, room_width, 4), dtype=torch.bool, device=device)
        position_transition_table = torch.zeros((room_height, room_width, 4, 2), dtype=torch.int64, device=device)
        
        # Create coordinate grids for vectorized operations
        row_grid, col_grid = torch.meshgrid(
            torch.arange(room_height, device=device),
            torch.arange(room_width, device=device),
            indexing='ij'
        )
        
        # Initialize transition table with current positions (default: no movement)
        for action_idx in range(4):
            position_transition_table[:, :, action_idx, 0] = row_grid
            position_transition_table[:, :, action_idx, 1] = col_grid
        
        # Action direction vectors: [up, down, left, right]
        action_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Transfer room layout to device
        room_on_device = self.room.to(device)
        
        # Convert invisible wall constraints to device tensors for fast lookup
        invisible_wall_masks = self._build_invisible_wall_masks(device, room_height, room_width)
        
        # Process each action type
        for action_idx, (delta_row, delta_col) in enumerate(action_directions):
            
            # Compute potential new positions
            new_row_positions = row_grid + delta_row
            new_col_positions = col_grid + delta_col
            
            # Check boundary constraints
            within_room_bounds = (
                (new_row_positions >= 0) & (new_row_positions < room_height) &
                (new_col_positions >= 0) & (new_col_positions < room_width)
            )
            
            # Check room layout constraints (walls)
            no_physical_walls = torch.ones_like(within_room_bounds, dtype=torch.bool)
            valid_positions_mask = within_room_bounds
            no_physical_walls[valid_positions_mask] = (
                room_on_device[new_row_positions[valid_positions_mask], 
                             new_col_positions[valid_positions_mask]] != -1
            )
            
            # Check invisible wall constraints
            position_flat_indices = row_grid * room_width + col_grid
            no_invisible_walls = ~invisible_wall_masks[action_idx][position_flat_indices]
            
            # Combine all constraints
            action_is_valid = within_room_bounds & no_physical_walls & no_invisible_walls
            action_validity_table[:, :, action_idx] = action_is_valid
            
            # Update transition table for valid actions
            self._update_transition_table_for_action(
                position_transition_table, action_idx, action_is_valid,
                new_row_positions, new_col_positions
            )
        
        return action_validity_table, position_transition_table
    
    def _build_invisible_wall_masks(self, device: torch.device, 
                                  room_height: int, room_width: int) -> list[torch.Tensor]:
        """Build GPU tensors for invisible wall constraints."""
        total_positions = room_height * room_width
        
        invisible_wall_tensors = []
        for wall_set in [self.no_up, self.no_down, self.no_left, self.no_right]:
            wall_mask = torch.zeros(total_positions, dtype=torch.bool, device=device)
            if wall_set:
                wall_indices = torch.tensor(list(wall_set), device=device, dtype=torch.long)
                wall_mask[wall_indices] = True
            invisible_wall_tensors.append(wall_mask)
        
        return invisible_wall_tensors
    
    def _update_transition_table_for_action(self, transition_table: torch.Tensor, 
                                          action_idx: int, validity_mask: torch.Tensor,
                                          new_rows: torch.Tensor, new_cols: torch.Tensor) -> None:
        """Update position transition table for valid actions."""
        valid_position_indices = validity_mask.nonzero(as_tuple=False)
        
        if len(valid_position_indices) > 0:
            valid_rows = valid_position_indices[:, 0]
            valid_cols = valid_position_indices[:, 1]
            
            transition_table[valid_rows, valid_cols, action_idx, 0] = new_rows[valid_rows, valid_cols]
            transition_table[valid_rows, valid_cols, action_idx, 1] = new_cols[valid_rows, valid_cols]
    
    def _process_sequence_chunk(self, chunk_start: int, chunk_end: int,
                              lookup_tables: dict, sequence_tensors: dict,
                              device: torch.device) -> None:
        """
        Process a chunk of the navigation sequence using GPU-optimized operations.
        
        This method processes sequence steps sequentially within each chunk because
        each step depends on the previous position. However, it minimizes CPU-GPU
        synchronization by keeping all operations on the target device.
        
        Args:
            chunk_start: Starting index of the chunk (inclusive)
            chunk_end: Ending index of the chunk (exclusive)
            lookup_tables: Pre-computed lookup tables for observations and transitions
            sequence_tensors: Allocated tensors for storing sequence data
            device: Target device for computation
        """
        observations = sequence_tensors['observations']
        actions = sequence_tensors['actions']
        positions = sequence_tensors['positions']
        random_actions = sequence_tensors['random_actions']
        
        observation_table = lookup_tables['obs_table']
        action_validity_table = lookup_tables['valid_actions_table']
        position_transition_table = lookup_tables['transition_table']
        
        # Process each time step in the chunk
        for time_step in range(chunk_start, chunk_end):
            if time_step == 0:
                continue  # Initial position already set
            
            # Get current position (maintain as GPU tensors to avoid synchronization)
            current_position = positions[time_step - 1]  # Shape: [2]
            current_row, current_col = current_position[0], current_position[1]
            
            # Look up observation for current position using pre-computed table
            observations[time_step] = observation_table[current_row, current_col]
            
            # Get randomly proposed action for this time step
            proposed_action = random_actions[time_step]
            
            # Validate proposed action using pre-computed validity table
            action_is_valid = action_validity_table[current_row, current_col, proposed_action]
            
            # Select final action (proposed if valid, otherwise find valid alternative)
            if action_is_valid:
                selected_action = proposed_action
            else:
                # Find first valid action using vectorized search
                available_actions_mask = action_validity_table[current_row, current_col]
                if available_actions_mask.any():
                    valid_action_indices = available_actions_mask.nonzero(as_tuple=True)[0]
                    selected_action = valid_action_indices[0]
                else:
                    # Fallback: use action 0 (should rarely happen in well-formed environments)
                    selected_action = torch.tensor(0, device=device, dtype=torch.int64)
            
            actions[time_step] = selected_action
            
            # Update position using pre-computed transition table
            positions[time_step] = position_transition_table[current_row, current_col, selected_action]
    
    
def save_room_plot(room, save_path_base, cmap='viridis'):
    """
    Save room plot as both PDF and PNG.
    
    Args:
        room (array-like): Room layout data
        save_pa
        th_base (str): Base path for saving (without extension)
        cmap (str or colormap): Colormap to use for plotting
    """
    # Input validation
    assert room is not None, "room cannot be None"
    assert isinstance(save_path_base, str), f"save_path_base must be str, got {type(save_path_base)}"
    assert len(save_path_base) > 0, "save_path_base cannot be empty"
    
    # Convert to numpy if needed
    if isinstance(room, torch.Tensor):
        if room.is_cuda:
            room = room.cpu()
        room = room.numpy()
    
    room = np.asarray(room)
    assert room.ndim == 2, f"room must be 2D, got {room.ndim}D"
    assert room.shape[0] > 0 and room.shape[1] > 0, f"room must have positive dimensions, got {room.shape}"
    
    # Create and save plot
    plt.figure(figsize=(8, 6))
    plt.imshow(room, cmap=cmap)
    plt.colorbar()
    plt.title("Room Layout")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis('off')
    plt.tight_layout()
    
    # Save both formats
    plt.savefig(f"{save_path_base}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path_base}.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Room plot saved to {save_path_base}.pdf and {save_path_base}.png")
