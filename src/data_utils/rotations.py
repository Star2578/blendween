"""
6D Continuous Rotation Representation

Implementation based on:
"On the Continuity of Rotation Representations in Neural Networks"
Zhou et al., CVPR 2019

The 6D representation uses the first two columns of a 3x3 rotation matrix,
providing a continuous, unique representation without discontinuities.
"""

import numpy as np
import torch


def quat_to_6d_numpy(quat):
    """
    Convert quaternion to 6D continuous representation (NumPy).

    Args:
        quat: (..., 4) quaternion in [w, x, y, z] format

    Returns:
        rot6d: (..., 6) first two columns of rotation matrix flattened

    Note:
        Quaternion is normalized before conversion to ensure valid rotation.
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Convert to rotation matrix
    # First column
    r00 = 1 - 2*(y**2 + z**2)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    # Second column
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x**2 + z**2)
    r21 = 2*(y*z + w*x)

    # Stack to create 6D representation (first two columns)
    rot6d = np.stack([r00, r10, r20, r01, r11, r21], axis=-1)

    return rot6d


def quat_to_6d_torch(quat):
    """
    Convert quaternion to 6D continuous representation (PyTorch).

    Args:
        quat: (..., 4) quaternion tensor in [w, x, y, z] format

    Returns:
        rot6d: (..., 6) first two columns of rotation matrix flattened
    """
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Convert to rotation matrix
    # First column
    r00 = 1 - 2*(y**2 + z**2)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    # Second column
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x**2 + z**2)
    r21 = 2*(y*z + w*x)

    # Stack to create 6D representation
    rot6d = torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)

    return rot6d


def rotation_6d_to_matrix_numpy(rot6d):
    """
    Convert 6D representation to full 3x3 rotation matrix (NumPy).
    Uses Gram-Schmidt orthogonalization to recover the third column.

    Args:
        rot6d: (..., 6) first two columns of rotation matrix

    Returns:
        rotmat: (..., 3, 3) full rotation matrix
    """
    batch_shape = rot6d.shape[:-1]

    # Extract first two columns and reshape
    col1 = rot6d[..., 0:3]  # First column
    col2 = rot6d[..., 3:6]  # Second column

    # Normalize first column with stability check
    col1_norm = np.linalg.norm(col1, axis=-1, keepdims=True)
    col1_norm = np.maximum(col1_norm, 1e-8)  # Prevent division by zero
    col1 = col1 / col1_norm

    # Gram-Schmidt: make col2 orthogonal to col1
    col2 = col2 - np.sum(col1 * col2, axis=-1, keepdims=True) * col1

    # Normalize with stability check
    col2_norm = np.linalg.norm(col2, axis=-1, keepdims=True)
    col2_norm = np.maximum(col2_norm, 1e-8)  # Prevent division by zero
    col2 = col2 / col2_norm

    # Third column is cross product
    col3 = np.cross(col1, col2, axis=-1)

    # Stack into matrix (..., 3, 3)
    rotmat = np.stack([col1, col2, col3], axis=-1)

    return rotmat


def rotation_6d_to_matrix_torch(rot6d):
    """
    Convert 6D representation to full 3x3 rotation matrix (PyTorch).
    Uses Gram-Schmidt orthogonalization to recover the third column.

    Args:
        rot6d: (..., 6) first two columns of rotation matrix

    Returns:
        rotmat: (..., 3, 3) full rotation matrix
    """
    # Extract first two columns
    col1 = rot6d[..., 0:3]  # First column
    col2 = rot6d[..., 3:6]  # Second column

    # Normalize first column with stability check
    col1_norm = torch.norm(col1, dim=-1, keepdim=True)
    col1_norm = torch.clamp(col1_norm, min=1e-8)  # Prevent division by zero
    col1 = col1 / col1_norm

    # Gram-Schmidt: make col2 orthogonal to col1
    col2 = col2 - torch.sum(col1 * col2, dim=-1, keepdim=True) * col1

    # Normalize with stability check
    col2_norm = torch.norm(col2, dim=-1, keepdim=True)
    col2_norm = torch.clamp(col2_norm, min=1e-8)  # Prevent division by zero
    col2 = col2 / col2_norm

    # Third column is cross product
    col3 = torch.cross(col1, col2, dim=-1)

    # Stack into matrix (..., 3, 3)
    rotmat = torch.stack([col1, col2, col3], dim=-1)

    return rotmat


def rotation_6d_to_quat_numpy(rot6d):
    """
    Convert 6D representation to quaternion (NumPy).

    Args:
        rot6d: (..., 6) first two columns of rotation matrix

    Returns:
        quat: (..., 4) quaternion in [w, x, y, z] format
    """
    # First convert to rotation matrix
    rotmat = rotation_6d_to_matrix_numpy(rot6d)

    # Convert rotation matrix to quaternion
    # Using Shepperd's method for numerical stability
    trace = rotmat[..., 0, 0] + rotmat[..., 1, 1] + rotmat[..., 2, 2]

    # Allocate output
    batch_shape = rot6d.shape[:-1]
    quat = np.zeros(batch_shape + (4,))

    # Case 1: trace > 0
    mask1 = trace > 0
    s = np.sqrt(trace[mask1] + 1.0) * 2
    quat[mask1, 0] = 0.25 * s
    quat[mask1, 1] = (rotmat[mask1, 2, 1] - rotmat[mask1, 1, 2]) / s
    quat[mask1, 2] = (rotmat[mask1, 0, 2] - rotmat[mask1, 2, 0]) / s
    quat[mask1, 3] = (rotmat[mask1, 1, 0] - rotmat[mask1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (rotmat[..., 0, 0] > rotmat[..., 1, 1]) & (rotmat[..., 0, 0] > rotmat[..., 2, 2])
    s = np.sqrt(1.0 + rotmat[mask2, 0, 0] - rotmat[mask2, 1, 1] - rotmat[mask2, 2, 2]) * 2
    quat[mask2, 0] = (rotmat[mask2, 2, 1] - rotmat[mask2, 1, 2]) / s
    quat[mask2, 1] = 0.25 * s
    quat[mask2, 2] = (rotmat[mask2, 0, 1] + rotmat[mask2, 1, 0]) / s
    quat[mask2, 3] = (rotmat[mask2, 0, 2] + rotmat[mask2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (rotmat[..., 1, 1] > rotmat[..., 2, 2])
    s = np.sqrt(1.0 + rotmat[mask3, 1, 1] - rotmat[mask3, 0, 0] - rotmat[mask3, 2, 2]) * 2
    quat[mask3, 0] = (rotmat[mask3, 0, 2] - rotmat[mask3, 2, 0]) / s
    quat[mask3, 1] = (rotmat[mask3, 0, 1] + rotmat[mask3, 1, 0]) / s
    quat[mask3, 2] = 0.25 * s
    quat[mask3, 3] = (rotmat[mask3, 1, 2] + rotmat[mask3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = np.sqrt(1.0 + rotmat[mask4, 2, 2] - rotmat[mask4, 0, 0] - rotmat[mask4, 1, 1]) * 2
    quat[mask4, 0] = (rotmat[mask4, 1, 0] - rotmat[mask4, 0, 1]) / s
    quat[mask4, 1] = (rotmat[mask4, 0, 2] + rotmat[mask4, 2, 0]) / s
    quat[mask4, 2] = (rotmat[mask4, 1, 2] + rotmat[mask4, 2, 1]) / s
    quat[mask4, 3] = 0.25 * s

    return quat


# Convenience functions that auto-detect input type
def quat_to_6d(quat):
    """Auto-detect NumPy or PyTorch and convert quaternion to 6D."""
    if isinstance(quat, np.ndarray):
        return quat_to_6d_numpy(quat)
    else:
        return quat_to_6d_torch(quat)


def rotation_6d_to_matrix(rot6d):
    """Auto-detect NumPy or PyTorch and convert 6D to rotation matrix."""
    if isinstance(rot6d, np.ndarray):
        return rotation_6d_to_matrix_numpy(rot6d)
    else:
        return rotation_6d_to_matrix_torch(rot6d)
