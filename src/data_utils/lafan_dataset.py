"""
LAFAN1 Dataset Loader for SILK

PyTorch Dataset class that loads LAFAN1 motion capture data and prepares it
for SILK training with:
- Root-space feature extraction
- Zero-filling for missing frames
- Normalization
- Variable transition lengths
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from src.external.lafan1 import extract, utils

from .features import extract_root_space_features_numpy, get_feature_dims


class LAFANDataset(Dataset):
    """
    LAFAN1 dataset for SILK motion in-betweening.

    Loads motion sequences and prepares them for training with:
    - Sliding window extraction (offset=5 for SILK)
    - Root-space feature representation
    - Normalization using training statistics
    - Zero-filling for transition frames

    Args:
        bvh_folder: Path to folder containing BVH files
        actors: List of actor names to include (e.g., ['subject1', 'subject2'])
        window_size: Total sequence length (default: 50 for training)
        offset: Sliding window offset (default: 5 for SILK, 20 for baseline)
        context_frames: Number of context frames before transition (default: 10)
        train_stats: Dict with 'mean' and 'std' for normalization (None = compute from data)
    """

    def __init__(
        self,
        bvh_folder,
        actors,
        window_size=50,
        offset=5,
        context_frames=10,
        train_stats=None,
        cache_path=None
    ):
        super().__init__()

        self.bvh_folder = bvh_folder
        self.actors = actors
        self.window_size = window_size
        self.offset = offset
        self.context_frames = context_frames
        self.cache_path = cache_path

        print(f"Loading LAFAN1 dataset...")
        print(f"  Actors: {actors}")
        print(f"  Window size: {window_size}")
        print(f"  Offset: {offset}")
        print(f"  Context frames: {context_frames}")

        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading from cache: {cache_path}")
            self._load_from_cache(cache_path)
        else:
            # Load all sequences
            self.sequences = []
            self.skeleton_info = None  # Will store parents, offsets, bone names

            self._load_sequences()

            # Save to cache if path provided
            if cache_path:
                print(f"  Saving to cache: {cache_path}")
                self._save_to_cache(cache_path)

        # Get feature dimensions
        num_joints = len(self.skeleton_info['bones'])
        self.din, self.dout = get_feature_dims(num_joints)

        print(f"\nDataset loaded:")
        print(f"  Total sequences: {len(self.sequences)}")
        print(f"  Joints: {num_joints}")
        print(f"  Input features (din): {self.din}")
        print(f"  Output features (dout): {self.dout}")

        # Compute or load normalization statistics
        if train_stats is None:
            print(f"\nComputing normalization statistics...")
            self.stats = self._compute_statistics()
        else:
            print(f"\nUsing provided normalization statistics")
            self.stats = train_stats

    def _load_sequences(self):
        """Load all BVH files and extract sequences with sliding window."""
        bvh_files = sorted([f for f in os.listdir(self.bvh_folder) if f.endswith('.bvh')])

        for bvh_file in bvh_files:
            # Check if file belongs to one of our actors
            if not any(actor in bvh_file for actor in self.actors):
                continue

            file_path = os.path.join(self.bvh_folder, bvh_file)
            print(f"  Loading: {bvh_file}")

            # Read BVH file
            anim = extract.read_bvh(file_path)

            # Store skeleton info (same for all files)
            if self.skeleton_info is None:
                self.skeleton_info = {
                    'parents': anim.parents,
                    'offsets': anim.offsets,
                    'bones': anim.bones
                }

            # Extract windows with sliding offset
            num_frames = anim.quats.shape[0]

            i = 0
            while i + self.window_size < num_frames:
                # Extract window (MUST COPY to avoid modifying original data!)
                local_quats = anim.quats[i:i + self.window_size].copy()
                local_pos = anim.pos[i:i + self.window_size].copy()

                # ===== CRITICAL PREPROCESSING (matching LAFAN1 baseline) =====
                # First, compute global positions for centering
                global_quats_temp, global_pos_temp = utils.quat_fk(
                    local_quats,
                    local_pos,
                    anim.parents
                )

                # Center sequence around XZ = 0 (on LOCAL positions)
                # This prevents root positions from being at absurd coordinates
                root_xz_mean = np.mean(global_pos_temp[:, 0, [0, 2]], axis=0, keepdims=True)
                local_pos[:, 0, 0] = local_pos[:, 0, 0] - root_xz_mean[0, 0]
                local_pos[:, 0, 2] = local_pos[:, 0, 2] - root_xz_mean[0, 1]

                # Unify facing direction at last context frame (frame 10)
                # This makes the model rotation-invariant
                # rotate_at_frame expects and returns LOCAL pos/quat, does FK/IK internally
                local_pos_batch = local_pos[np.newaxis, ...]  # Add batch dim
                local_quats_batch = local_quats[np.newaxis, ...]

                local_pos_batch, local_quats_batch = utils.rotate_at_frame(
                    local_pos_batch,
                    local_quats_batch,
                    anim.parents,
                    n_past=self.context_frames
                )

                local_pos = local_pos_batch[0]  # Remove batch dim
                local_quats = local_quats_batch[0]
                # ===== END PREPROCESSING =====

                # Now compute final global positions and quaternions via FK
                global_quats, global_pos = utils.quat_fk(
                    local_quats,
                    local_pos,
                    anim.parents
                )

                # Extract root-space features
                # Input features (with velocities)
                input_features = extract_root_space_features_numpy(
                    global_quats,
                    global_pos,
                    include_velocity=True
                )

                # Output features (without velocities)
                output_features = extract_root_space_features_numpy(
                    global_quats,
                    global_pos,
                    include_velocity=False
                )

                # Store sequence
                self.sequences.append({
                    'input': input_features,   # (window_size, din)
                    'output': output_features,  # (window_size, dout)
                    'file': bvh_file,
                    'start_frame': i
                })

                i += self.offset

    def _save_to_cache(self, cache_path):
        """Save preprocessed sequences and skeleton info to cache file."""
        cache_data = {
            'sequences': self.sequences,
            'skeleton_info': self.skeleton_info,
            'bvh_folder': self.bvh_folder,
            'actors': self.actors,
            'window_size': self.window_size,
            'offset': self.offset,
            'context_frames': self.context_frames
        }

        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"  Cached {len(self.sequences)} sequences to {cache_path}")

    def _load_from_cache(self, cache_path):
        """Load preprocessed sequences and skeleton info from cache file."""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        self.sequences = cache_data['sequences']
        self.skeleton_info = cache_data['skeleton_info']

        # Verify cache matches current configuration
        if (cache_data['bvh_folder'] != self.bvh_folder or
            cache_data['actors'] != self.actors or
            cache_data['window_size'] != self.window_size or
            cache_data['offset'] != self.offset):
            print(f"  Warning: Cache configuration mismatch!")
            print(f"    Cached: {cache_data['actors']}, window={cache_data['window_size']}, offset={cache_data['offset']}")
            print(f"    Current: {self.actors}, window={self.window_size}, offset={self.offset}")

        print(f"  Loaded {len(self.sequences)} sequences from cache")

    def _compute_statistics(self):
        """Compute mean and std for normalization across all sequences."""
        # Collect all input features
        all_inputs = np.concatenate([seq['input'] for seq in self.sequences], axis=0)
        all_outputs = np.concatenate([seq['output'] for seq in self.sequences], axis=0)

        # Compute mean and standard deviation
        input_mean = np.mean(all_inputs, axis=0)
        input_std = np.std(all_inputs, axis=0)
        output_mean = np.mean(all_outputs, axis=0)
        output_std = np.std(all_outputs, axis=0)

        # CRITICAL FIX: Replace zero-variance dimensions with 1.0 instead of adding epsilon
        # Adding epsilon causes massive scaling (10^8) for near-constant features
        input_std = np.where(input_std < 1e-6, 1.0, input_std)
        output_std = np.where(output_std < 1e-6, 1.0, output_std)

        stats = {
            'input_mean': input_mean,
            'input_std': input_std,
            'output_mean': output_mean,
            'output_std': output_std
        }

        print(f"  Input mean shape: {stats['input_mean'].shape}")
        print(f"  Input std shape: {stats['input_std'].shape}")

        # Compute position-only normalization statistics for L2P metric
        # CRITICAL FIX: Must compute stats from GLOBAL positions, not root-space positions!
        # The L2P metric normalizes global positions during evaluation, so stats must match.
        print(f"\n  Computing GLOBAL position normalization statistics for L2P metric...")
        print(f"  (Processing {len(self.sequences)} sequences...)")

        num_joints = len(self.skeleton_info['bones'])

        # FAST VERSION: Reconstruct global positions directly without quaternion conversion
        # Output features: [root_xz(2), root_orient(2), joint_pos_rel(J*3), joint_rot_6d(J*6)]
        all_positions = []

        for seq in self.sequences:
            output_features = seq['output']  # (window_size, dout)

            # Extract root XZ position (first 2 dims)
            root_xz = output_features[:, 0:2]  # (window_size, 2)

            # Extract joint positions relative to root (dims 4 to 4+num_joints*3)
            joint_pos_start = 4
            joint_pos_end = 4 + num_joints * 3
            joint_pos_rel = output_features[:, joint_pos_start:joint_pos_end]  # (window_size, J*3)
            joint_pos_rel = joint_pos_rel.reshape(output_features.shape[0], num_joints, 3)

            # Reconstruct global positions by adding back root ground position
            # Root position on ground: (root_xz[0], 0, root_xz[1])
            root_pos_ground = np.zeros((output_features.shape[0], 1, 3))
            root_pos_ground[:, 0, 0] = root_xz[:, 0]  # X
            root_pos_ground[:, 0, 1] = 0              # Y = 0 (ground)
            root_pos_ground[:, 0, 2] = root_xz[:, 1]  # Z

            # Global positions = relative positions + root ground position
            global_pos = joint_pos_rel + root_pos_ground  # (window_size, num_joints, 3)

            # Flatten to (window_size, num_joints*3) for stats computation
            global_pos_flat = global_pos.reshape(output_features.shape[0], -1)
            all_positions.append(global_pos_flat)

        # Concatenate all: (total_frames, num_joints*3)
        all_positions = np.concatenate(all_positions, axis=0)
        print(f"  Collected {all_positions.shape[0]} total frames")

        # Transpose to (num_joints*3, total_frames) for computing stats
        positions_flat = all_positions.T

        # Compute mean and std with shape (1, num_joints*3, 1) for broadcasting
        x_mean = np.mean(positions_flat, axis=1, keepdims=True).reshape(1, -1, 1)
        x_std = np.std(positions_flat, axis=1, keepdims=True).reshape(1, -1, 1)

        # Prevent division by zero: use 1.0 for dimensions with zero variance
        # (These are constant dimensions like centered root X/Z coordinates)
        x_std = np.where(x_std < 1e-6, 1.0, x_std)

        stats['x_mean'] = x_mean
        stats['x_std'] = x_std

        print(f"  ✓ Position mean shape: {stats['x_mean'].shape}")
        print(f"  ✓ Position std shape: {stats['x_std'].shape}")
        print(f"  Sample x_mean (first 3): {stats['x_mean'][0, :3, 0]}")
        print(f"  Sample x_std (first 3): {stats['x_std'][0, :3, 0]}")

        return stats

    def __len__(self):
        """Return number of sequences in dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a single training sample.

        Returns:
            dict with keys:
                - 'input': (window_size, din) input features (normalized)
                - 'output': (window_size, dout) output features (normalized)
                - 'context_frames': int, number of context frames
        """
        seq = self.sequences[idx]

        # Get features
        input_features = seq['input'].copy()  # (window_size, din)
        output_features = seq['output'].copy()  # (window_size, dout)

        # Normalize
        input_features = (input_features - self.stats['input_mean']) / self.stats['input_std']
        output_features = (output_features - self.stats['output_mean']) / self.stats['output_std']

        # Convert to torch tensors
        input_features = torch.from_numpy(input_features).float()
        output_features = torch.from_numpy(output_features).float()

        return {
            'input': input_features,
            'output': output_features,
            'context_frames': self.context_frames
        }

    def save_statistics(self, filepath):
        """Save normalization statistics to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.stats, f)
        print(f"Statistics saved to: {filepath}")

    @staticmethod
    def load_statistics(filepath):
        """Load normalization statistics from file."""
        with open(filepath, 'rb') as f:
            stats = pickle.load(f)
        print(f"Statistics loaded from: {filepath}")
        return stats


class PreloadedSILKDataset(Dataset):
    """
    Preloaded SILK dataset - all data preprocessed and kept in RAM as tensors.

    This dataset:
    - Normalizes all sequences upfront and converts to PyTorch tensors
    - Stores everything in RAM (requires ~2-4GB for full LAFAN1)
    - Only does zero-filling and masking in __getitem__ (very fast)
    - Eliminates all disk I/O and numpy operations during training

    Trade-off: Uses more RAM but provides maximum throughput.
    Recommended when you have plenty of RAM (>= 16GB).

    Args:
        base_dataset: LAFANDataset instance
        min_transition: Minimum transition length
        max_transition: Maximum transition length
        context_frames: Number of context frames
    """

    def __init__(
        self,
        base_dataset,
        min_transition=5,
        max_transition=30,
        context_frames=10
    ):
        super().__init__()

        self.min_transition = min_transition
        self.max_transition = max_transition
        self.context_frames = context_frames

        print(f"\nPreloading dataset to RAM...")
        print(f"  Base sequences: {len(base_dataset)}")
        print(f"  Transition lengths: {min_transition}-{max_transition}")

        # Preload and normalize all sequences as tensors
        self.inputs = []
        self.targets = []

        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            # Data is already normalized and converted to tensors by base dataset
            self.inputs.append(sample['input'])    # (window_size, din)
            self.targets.append(sample['output'])  # (window_size, dout)

        # Estimate memory usage
        total_input_size = sum(t.numel() * t.element_size() for t in self.inputs)
        total_target_size = sum(t.numel() * t.element_size() for t in self.targets)
        total_mb = (total_input_size + total_target_size) / (1024 * 1024)

        print(f"  Preloaded {len(self.inputs)} sequences")
        print(f"  Memory usage: ~{total_mb:.1f} MB")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get a training sample with zero-filling.

        All normalization and tensor conversion already done - just need
        zero-filling and mask creation which are very fast operations.
        """
        # Get preloaded tensors (no copy needed, PyTorch handles this)
        input_features = self.inputs[idx]   # (window_size, din)
        output_features = self.targets[idx]  # (window_size, dout)

        # Randomly sample transition length
        transition_length = np.random.randint(
            self.min_transition,
            min(self.max_transition + 1, len(input_features) - self.context_frames - 1)
        )

        # Create sequence: [context frames] + [transition frames] + [target keyframe]
        seq_len = self.context_frames + transition_length + 1

        # Extract subsequence
        input_seq = input_features[:seq_len].clone()   # (seq_len, din)
        target_seq = output_features[:seq_len].clone()  # (seq_len, dout)

        # Zero-fill transition frames in input
        input_seq[self.context_frames:seq_len-1] = 0.0

        # Create mask: True for frames we want to predict (transition frames only)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[self.context_frames:seq_len-1] = True

        return {
            'input': input_seq,           # (seq_len, din) with zeros
            'target': target_seq,         # (seq_len, dout) ground truth
            'mask': mask,                 # (seq_len,) predict these frames
            'transition_length': transition_length
        }


class SILKTrainingDataset(Dataset):
    """
    SILK training dataset with zero-filling and variable transition lengths.

    Wraps LAFANDataset and applies SILK-specific processing:
    - Zero-fills transition frames
    - Samples variable transition lengths (5-30 frames during training)
    - Prepares input in SILK format: [context, zeros, target_keyframe]

    Args:
        base_dataset: LAFANDataset instance
        min_transition: Minimum transition length (default: 5)
        max_transition: Maximum transition length (default: 30)
        context_frames: Number of context frames (default: 10)
    """

    def __init__(
        self,
        base_dataset,
        min_transition=5,
        max_transition=30,
        context_frames=10
    ):
        super().__init__()

        self.base_dataset = base_dataset
        self.min_transition = min_transition
        self.max_transition = max_transition
        self.context_frames = context_frames

        print(f"\nSILK Training Dataset:")
        print(f"  Base sequences: {len(base_dataset)}")
        print(f"  Transition lengths: {min_transition}-{max_transition}")
        print(f"  Context frames: {context_frames}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Get a training sample with zero-filling.

        Returns:
            dict with keys:
                - 'input': (seq_len, din) with zero-filled transition
                - 'target': (seq_len, dout) ground truth
                - 'mask': (seq_len,) boolean mask (True = predict this frame)
                - 'transition_length': int, length of transition
        """
        # Get base sequence
        sample = self.base_dataset[idx]
        input_features = sample['input']  # (window_size, din)
        output_features = sample['output']  # (window_size, dout)

        # Randomly sample transition length
        transition_length = np.random.randint(
            self.min_transition,
            min(self.max_transition + 1, len(input_features) - self.context_frames - 1)
        )

        # Create sequence: [context frames] + [transition frames] + [target keyframe]
        seq_len = self.context_frames + transition_length + 1

        # Extract subsequence
        input_seq = input_features[:seq_len].clone()  # (seq_len, din)
        target_seq = output_features[:seq_len].clone()  # (seq_len, dout)

        # Zero-fill transition frames in input
        # Keep: context frames (0 to context_frames-1) and target keyframe (seq_len-1)
        # Zero: transition frames (context_frames to seq_len-2)
        input_seq[self.context_frames:seq_len-1] = 0.0

        # Create mask: True for frames we want to predict (transition frames only)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[self.context_frames:seq_len-1] = True

        return {
            'input': input_seq,           # (seq_len, din) with zeros
            'target': target_seq,         # (seq_len, dout) ground truth
            'mask': mask,                 # (seq_len,) predict these frames
            'transition_length': transition_length
        }


def collate_fn_variable_length(batch):
    """
    Custom collate function for variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Find max sequence length in batch
    max_len = max(sample['input'].shape[0] for sample in batch)

    batch_size = len(batch)
    din = batch[0]['input'].shape[1]
    dout = batch[0]['target'].shape[1]

    # Initialize padded tensors
    input_padded = torch.zeros(batch_size, max_len, din)
    target_padded = torch.zeros(batch_size, max_len, dout)
    mask_padded = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    transition_lengths = []

    # Fill in data
    for i, sample in enumerate(batch):
        seq_len = sample['input'].shape[0]
        input_padded[i, :seq_len] = sample['input']
        target_padded[i, :seq_len] = sample['target']
        mask_padded[i, :seq_len] = sample['mask']
        lengths[i] = seq_len
        transition_lengths.append(sample['transition_length'])

    return {
        'input': input_padded,
        'target': target_padded,
        'mask': mask_padded,
        'lengths': lengths,
        'transition_length': torch.tensor(transition_lengths)
    }


def create_dataloaders(
    bvh_folder,
    train_actors,
    test_actors,
    batch_size=64,
    window_size=50,
    offset=5,
    context_frames=10,
    min_transition=5,
    max_transition=30,
    num_workers=4,
    cache_dir='data/cache',
    preload=False
):
    """
    Create train and test dataloaders for SILK.

    Args:
        bvh_folder: Path to BVH files
        train_actors: List of training actor names
        test_actors: List of test actor names
        batch_size: Batch size for training
        window_size: Sequence window size
        offset: Sliding window offset
        context_frames: Number of context frames
        min_transition: Min transition length for training
        max_transition: Max transition length for training
        num_workers: Number of dataloader workers
        cache_dir: Directory to store cache files (None to disable caching)
        preload: If True, use PreloadedSILKDataset (faster but uses more RAM)

    Returns:
        (train_loader, test_loader, train_stats)
    """
    # Generate cache paths based on dataset configuration
    train_cache_path = None
    test_cache_path = None
    stats_cache_path = None

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

        # Create unique cache filenames based on configuration
        train_actors_str = '_'.join(sorted(train_actors))
        test_actors_str = '_'.join(sorted(test_actors))

        train_cache_path = os.path.join(
            cache_dir,
            f'train_{train_actors_str}_w{window_size}_o{offset}.pkl'
        )
        test_cache_path = os.path.join(
            cache_dir,
            f'test_{test_actors_str}_w65_o40.pkl'
        )
        stats_cache_path = os.path.join(
            cache_dir,
            f'stats_{train_actors_str}_w{window_size}_o{offset}.pkl'
        )

    # Try to load cached statistics first
    train_stats = None
    if stats_cache_path and os.path.exists(stats_cache_path):
        print(f"Loading cached statistics from: {stats_cache_path}")
        train_stats = LAFANDataset.load_statistics(stats_cache_path)

    # Create training dataset
    print("=" * 60)
    print("Creating Training Dataset")
    print("=" * 60)
    train_base = LAFANDataset(
        bvh_folder=bvh_folder,
        actors=train_actors,
        window_size=window_size,
        offset=offset,
        context_frames=context_frames,
        train_stats=train_stats,  # Use cached stats if available
        cache_path=train_cache_path
    )

    # Save statistics to cache if not already cached
    if stats_cache_path and not os.path.exists(stats_cache_path):
        train_base.save_statistics(stats_cache_path)

    # Choose dataset type based on preload option
    dataset_class = PreloadedSILKDataset if preload else SILKTrainingDataset
    dataset_type_name = "Preloaded" if preload else "Standard"

    print(f"\nCreating {dataset_type_name} training dataset...")
    train_dataset = dataset_class(
        base_dataset=train_base,
        min_transition=min_transition,
        max_transition=max_transition,
        context_frames=context_frames
    )

    # Create test dataset (using training stats for normalization)
    print("\n" + "=" * 60)
    print("Creating Test Dataset")
    print("=" * 60)
    test_base = LAFANDataset(
        bvh_folder=bvh_folder,
        actors=test_actors,
        window_size=65,  # Longer for testing
        offset=40,  # Larger offset for test
        context_frames=context_frames,
        train_stats=train_base.stats,  # Use training stats!
        cache_path=test_cache_path
    )

    print(f"\nCreating {dataset_type_name} test dataset...")
    test_dataset = dataset_class(
        base_dataset=test_base,
        min_transition=5,  # Test on various lengths
        max_transition=45,
        context_frames=context_frames
    )

    # Create dataloaders with custom collate function
    # Configure DataLoader for optimal performance
    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': True,
        'collate_fn': collate_fn_variable_length,
    }

    # Add multiprocessing options if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs.update({
            'num_workers': num_workers,
            'persistent_workers': True,  # Keep workers alive between epochs
            'prefetch_factor': 2,         # Prefetch 2 batches per worker
        })
    else:
        dataloader_kwargs['num_workers'] = 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    print("\n" + "=" * 60)
    print("Dataloaders Created")
    print("=" * 60)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader, train_base.stats
