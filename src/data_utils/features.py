"""
Root-Space Feature Extraction for SILK

Implements SILK's key innovation: representing motion relative to the character's
root (hip) position and orientation, making the features translation and rotation
invariant.

Feature dimensions:
- Input (with velocities): din = 18J + 8
- Output (without velocities): dout = 9J + 4

where J is the number of joints (22 for LAFAN1).
"""

import numpy as np
import torch
from .rotations import quat_to_6d_numpy, quat_to_6d_torch


def extract_root_orient_2d_numpy(root_quat):
    """
    Extract 2D orientation (facing direction) from root quaternion (NumPy).

    Args:
        root_quat: (..., 4) root quaternion in [w, x, y, z] format

    Returns:
        orient_2d: (..., 2) [cos(yaw), sin(yaw)] representation
    """
    w, x, y, z = root_quat[..., 0], root_quat[..., 1], root_quat[..., 2], root_quat[..., 3]

    # Extract yaw angle (rotation around Y axis)
    yaw = np.arctan2(2*(w*y + x*z), 1 - 2*(y**2 + z**2))

    # Convert to [cos, sin] representation
    orient_2d = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)

    return orient_2d


def extract_root_orient_2d_torch(root_quat):
    """
    Extract 2D orientation (facing direction) from root quaternion (PyTorch).

    Args:
        root_quat: (..., 4) root quaternion in [w, x, y, z] format

    Returns:
        orient_2d: (..., 2) [cos(yaw), sin(yaw)] representation
    """
    w, x, y, z = root_quat[..., 0], root_quat[..., 1], root_quat[..., 2], root_quat[..., 3]

    # Extract yaw angle
    yaw = torch.atan2(2*(w*y + x*z), 1 - 2*(y**2 + z**2))

    # Convert to [cos, sin] representation
    orient_2d = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

    return orient_2d


def extract_root_space_features_numpy(global_quats, global_pos, include_velocity=True):
    """
    Extract SILK's root-space features from global positions and quaternions (NumPy).

    Args:
        global_quats: (frames, joints, 4) global quaternions [w, x, y, z]
        global_pos: (frames, joints, 3) global positions [x, y, z]
        include_velocity: If True, compute and include velocity features

    Returns:
        features: (frames, din) if include_velocity else (frames, dout)

    Feature breakdown:
        Root features:
            - 2D position (x, z on ground plane)
            - 2D orientation (cos(yaw), sin(yaw))
            - [Optional] 2D linear velocity
            - [Optional] 2D angular velocity

        Joint features (for all J joints):
            - 3D position (relative to root)
            - 6D rotation (continuous representation)
            - [Optional] 3D linear velocity
            - [Optional] 6D angular velocity
    """
    num_frames, num_joints, _ = global_pos.shape

    # Extract root (hip) information
    root_pos_3d = global_pos[:, 0, :]  # (frames, 3)
    root_quat = global_quats[:, 0, :]  # (frames, 4)

    # Root 2D position (XZ plane only, ignore height)
    root_pos_2d = root_pos_3d[:, [0, 2]]  # (frames, 2)

    # Root 2D orientation (facing direction)
    root_orient_2d = extract_root_orient_2d_numpy(root_quat)  # (frames, 2)

    # Joint positions relative to root (root-space)
    # Root space: relative to hip PROJECTION on ground (Y=0), not 3D hip position
    root_pos_ground = root_pos_3d.copy()
    root_pos_ground[:, 1] = 0  # Project to ground: set Y=0
    joints_pos_root = global_pos - root_pos_ground[:, None, :]  # (frames, joints, 3)
    joints_pos_root = joints_pos_root.reshape(num_frames, -1)  # (frames, joints*3)

    # Joint rotations in 6D representation
    joints_rot_6d = quat_to_6d_numpy(global_quats)  # (frames, joints, 6)
    joints_rot_6d = joints_rot_6d.reshape(num_frames, -1)  # (frames, joints*6)

    # Concatenate base features
    features = np.concatenate([
        root_pos_2d,      # (frames, 2)
        root_orient_2d,   # (frames, 2)
        joints_pos_root,  # (frames, joints*3)
        joints_rot_6d     # (frames, joints*6)
    ], axis=-1)

    if include_velocity:
        # Compute velocities using proper finite differences
        # FIX: Use forward/backward/central differences instead of zero-padding first frame

        def compute_velocity(pos):
            """Compute velocity with proper finite differences."""
            vel = np.zeros_like(pos)
            if len(pos) == 1:
                return vel  # Single frame, velocity is zero
            vel[0] = pos[1] - pos[0]  # Forward difference for first frame
            if len(pos) > 2:
                vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0  # Central difference for middle
            vel[-1] = pos[-1] - pos[-2]  # Backward difference for last frame
            return vel

        # Compute velocities for all components
        root_vel_2d = compute_velocity(root_pos_2d)
        root_ang_vel = compute_velocity(root_orient_2d)
        joints_lin_vel = compute_velocity(joints_pos_root)
        joints_ang_vel = compute_velocity(joints_rot_6d)

        # Add velocities to features
        features = np.concatenate([
            features,
            root_vel_2d,      # (frames, 2)
            root_ang_vel,     # (frames, 2)
            joints_lin_vel,   # (frames, joints*3)
            joints_ang_vel    # (frames, joints*6)
        ], axis=-1)

    return features


def extract_root_space_features_torch(global_quats, global_pos, include_velocity=True):
    """
    Extract SILK's root-space features from global positions and quaternions (PyTorch).

    Args:
        global_quats: (frames, joints, 4) global quaternions [w, x, y, z]
        global_pos: (frames, joints, 3) global positions [x, y, z]
        include_velocity: If True, compute and include velocity features

    Returns:
        features: (frames, din) if include_velocity else (frames, dout)
    """
    num_frames, num_joints, _ = global_pos.shape

    # Extract root information
    root_pos_3d = global_pos[:, 0, :]  # (frames, 3)
    root_quat = global_quats[:, 0, :]  # (frames, 4)

    # Root 2D position (XZ plane)
    root_pos_2d = root_pos_3d[:, [0, 2]]  # (frames, 2)

    # Root 2D orientation
    root_orient_2d = extract_root_orient_2d_torch(root_quat)  # (frames, 2)

    # Joint positions relative to root (root-space)
    # Root space: relative to hip PROJECTION on ground (Y=0), not 3D hip position
    root_pos_ground = root_pos_3d.clone()
    root_pos_ground[:, 1] = 0  # Project to ground: set Y=0
    joints_pos_root = global_pos - root_pos_ground[:, None, :]  # (frames, joints, 3)
    joints_pos_root = joints_pos_root.reshape(num_frames, -1)  # (frames, joints*3)

    # Joint rotations in 6D
    joints_rot_6d = quat_to_6d_torch(global_quats)  # (frames, joints, 6)
    joints_rot_6d = joints_rot_6d.reshape(num_frames, -1)  # (frames, joints*6)

    # Concatenate base features
    features = torch.cat([
        root_pos_2d,
        root_orient_2d,
        joints_pos_root,
        joints_rot_6d
    ], dim=-1)

    if include_velocity:
        # Compute velocities
        root_vel_2d = torch.zeros_like(root_pos_2d)
        root_vel_2d[1:] = root_pos_2d[1:] - root_pos_2d[:-1]

        root_ang_vel = torch.zeros_like(root_orient_2d)
        root_ang_vel[1:] = root_orient_2d[1:] - root_orient_2d[:-1]

        joints_lin_vel = torch.zeros_like(joints_pos_root)
        joints_lin_vel[1:] = joints_pos_root[1:] - joints_pos_root[:-1]

        joints_ang_vel = torch.zeros_like(joints_rot_6d)
        joints_ang_vel[1:] = joints_rot_6d[1:] - joints_rot_6d[:-1]

        # # Compute velocities using proper finite differences
        # # Same approach as NumPy version: forward/central/backward differences

        # def compute_velocity(pos):
        #     """Compute velocity with proper finite differences."""
        #     vel = torch.zeros_like(pos)
        #     if len(pos) == 1:
        #         return vel  # Single frame, velocity is zero
        #     vel[0] = pos[1] - pos[0]  # Forward difference for first frame
        #     if len(pos) > 2:
        #         vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0  # Central difference for middle
        #     vel[-1] = pos[-1] - pos[-2]  # Backward difference for last frame
        #     return vel

        # # Compute velocities for all components
        # root_vel_2d = compute_velocity(root_pos_2d)
        # root_ang_vel = compute_velocity(root_orient_2d)
        # joints_lin_vel = compute_velocity(joints_pos_root)
        # joints_ang_vel = compute_velocity(joints_rot_6d)

        # Add velocities
        features = torch.cat([
            features,
            root_vel_2d,
            root_ang_vel,
            joints_lin_vel,
            joints_ang_vel
        ], dim=-1)

    return features


def get_feature_dims(num_joints):
    """
    Get input and output feature dimensions for SILK.

    Args:
        num_joints: Number of joints in skeleton (22 for LAFAN1)

    Returns:
        (din, dout): Input and output feature dimensions
    """
    # Output: root (2 pos + 2 orient) + joints (3 pos + 6 rot) * J
    dout = 4 + 9 * num_joints

    # Input: output + velocities
    # Velocities: root (2 lin_vel + 2 ang_vel) + joints (3 lin_vel + 6 ang_vel) * J
    din = dout + 4 + 9 * num_joints

    return din, dout


# Convenience function that auto-detects input type
def extract_root_space_features(global_quats, global_pos, include_velocity=True):
    """
    Auto-detect NumPy or PyTorch and extract root-space features.

    Args:
        global_quats: (frames, joints, 4) global quaternions
        global_pos: (frames, joints, 3) global positions
        include_velocity: If True, include velocity features

    Returns:
        features: (frames, din) or (frames, dout)
    """
    if isinstance(global_pos, np.ndarray):
        return extract_root_space_features_numpy(global_quats, global_pos, include_velocity)
    else:
        return extract_root_space_features_torch(global_quats, global_pos, include_velocity)
