"""
Reconstruction from Root-Space Features

Converts SILK's root-space predictions back to global coordinates
for evaluation. This is the inverse of the feature extraction process.
"""

import numpy as np
import torch
import sys
import os

# Add LAFAN1 utilities
from src.external.lafan1 import utils

from src.data_utils.rotations import rotation_6d_to_quat_numpy


def reconstruct_global_from_root_space(root_space_features, num_joints=22):
    """
    Reconstruct global positions and quaternions from root-space features.

    This is the inverse of extract_root_space_features_numpy().

    Args:
        root_space_features: Array of shape (batch, timesteps, dout=202) or (timesteps, dout)
                            Contains: [root_2d_pos, root_2d_orient, joint_3d_pos, joint_6d_rot]
        num_joints: Number of joints (default: 22 for LAFAN1)

    Returns:
        global_quats: Global quaternions (batch, timesteps, joints, 4) or (timesteps, joints, 4)
        global_pos: Global positions (batch, timesteps, joints, 3) or (timesteps, joints, 3)
    """
    # Handle both batched and unbatched input
    if root_space_features.ndim == 2:
        # Add batch dimension
        root_space_features = root_space_features[np.newaxis, ...]
        unbatched = True
    else:
        unbatched = False

    # Convert to numpy if needed
    if torch.is_tensor(root_space_features):
        root_space_features = root_space_features.detach().cpu().numpy()

    batch, timesteps, _ = root_space_features.shape

    # Extract components from root-space features
    # Feature layout: [root_2d_pos (2), root_2d_orient (2), joint_3d_pos (J*3), joint_6d_rot (J*6)]
    idx = 0

    # Root 2D position (XZ plane)
    root_xz = root_space_features[:, :, idx:idx+2]  # (batch, timesteps, 2)
    idx += 2

    # Root 2D orientation (cos, sin of yaw angle)
    root_orient = root_space_features[:, :, idx:idx+2]  # (batch, timesteps, 2)
    idx += 2

    # Joint 3D positions (relative to root)
    joint_pos_flat = root_space_features[:, :, idx:idx+num_joints*3]  # (batch, timesteps, J*3)
    joint_pos_rel = joint_pos_flat.reshape(batch, timesteps, num_joints, 3)
    idx += num_joints * 3

    # Joint 6D rotations
    joint_rot_flat = root_space_features[:, :, idx:idx+num_joints*6]  # (batch, timesteps, J*6)
    joint_rot_6d = joint_rot_flat.reshape(batch, timesteps, num_joints, 6)

    # Reconstruct global positions and rotations
    # IMPORTANT: The joint positions stored in features are ALREADY in root-space
    # (relative to ground-projected hip), and the joint rotations are ALREADY global.
    # We just need to add back the ground projection position.

    # Root position on ground (Y=0 projection of hip)
    root_pos_ground = np.zeros((batch, timesteps, 1, 3))
    root_pos_ground[:, :, 0, 0] = root_xz[:, :, 0]  # X
    root_pos_ground[:, :, 0, 1] = 0                 # Y = 0 (ground projection)
    root_pos_ground[:, :, 0, 2] = root_xz[:, :, 1]  # Z

    # Global positions: joint_pos_rel is already relative to ground projection
    # Just add back the ground projection coordinates
    global_pos = joint_pos_rel + root_pos_ground  # (batch, timesteps, num_joints, 3)

    # Reconstruct global rotations from 6D representation
    # The rotations stored in features are ALREADY global rotations
    # (they were extracted directly from global_quats in feature extraction)
    global_quats = np.zeros((batch, timesteps, num_joints, 4))

    for b in range(batch):
        for t in range(timesteps):
            for j in range(num_joints):
                # Convert 6D to quaternion
                rot_6d = joint_rot_6d[b, t, j]
                quat = rotation_6d_to_quat_numpy(rot_6d)
                global_quats[b, t, j] = quat

    # Note: We do NOT need to multiply by root quaternion because the rotations
    # in the features are already global rotations (extracted from global_quats in features.py)

    # Remove batch dimension if input was unbatched
    if unbatched:
        global_quats = global_quats[0]
        global_pos = global_pos[0]

    return global_quats, global_pos


def denormalize_features(normalized_features, stats):
    """
    Denormalize features using training statistics.

    Args:
        normalized_features: Normalized features (batch, timesteps, dout) or (timesteps, dout)
        stats: Dict with 'output_mean' and 'output_std' arrays

    Returns:
        denormalized_features: Denormalized features (same shape as input)
    """
    # Convert to numpy if needed
    if torch.is_tensor(normalized_features):
        normalized_features = normalized_features.detach().cpu().numpy()

    output_mean = stats['output_mean']
    output_std = stats['output_std']

    # Denormalize: features * std + mean
    denormalized = normalized_features * output_std + output_mean

    return denormalized


def reconstruct_from_model_output(
    model_output,
    stats,
    num_joints=22
):
    """
    Complete pipeline: denormalize and reconstruct global coordinates.

    Args:
        model_output: Model predictions (batch, timesteps, dout) - normalized
        stats: Training statistics dict
        num_joints: Number of joints

    Returns:
        global_quats: Global quaternions (batch, timesteps, joints, 4)
        global_pos: Global positions (batch, timesteps, joints, 3)
    """
    # Denormalize
    denormalized = denormalize_features(model_output, stats)

    # Reconstruct global coordinates
    global_quats, global_pos = reconstruct_global_from_root_space(
        denormalized,
        num_joints=num_joints
    )

    return global_quats, global_pos


if __name__ == "__main__":
    # Test reconstruction
    print("Testing reconstruction from root-space features...")

    timesteps = 30
    num_joints = 22
    dout = 9 * num_joints + 4  # 202

    # Create dummy root-space features
    features = np.random.randn(timesteps, dout)

    # Reconstruct
    global_quats, global_pos = reconstruct_global_from_root_space(features, num_joints)

    print(f"\nInput shape: {features.shape}")
    print(f"Output global_quats shape: {global_quats.shape}")
    print(f"Output global_pos shape: {global_pos.shape}")

    # Test batched
    features_batch = np.random.randn(4, timesteps, dout)
    global_quats_batch, global_pos_batch = reconstruct_global_from_root_space(
        features_batch,
        num_joints
    )

    print(f"\nBatched input shape: {features_batch.shape}")
    print(f"Batched global_quats shape: {global_quats_batch.shape}")
    print(f"Batched global_pos shape: {global_pos_batch.shape}")

    print("\nReconstruction module working correctly!")
