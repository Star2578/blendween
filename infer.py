#!/usr/bin/env python3
"""
SILK Inference Script

Run SILK model on a single BVH file and output predictions to a new BVH file.

Usage:
    python infer.py --bvh input.bvh --start-context 0 --start-predict 10 --target 40 \
                    --checkpoint checkpoints/best_model.pth --output output.bvh

The script will:
    1. Load the BVH file
    2. Extract frames from start-context to target
    3. Zero-fill frames from start-predict to target-1 (transition frames)
    4. Run SILK inference to predict the transition
    5. Write the predicted motion to output BVH file
"""

import argparse
import yaml
import torch
import numpy as np
import pickle
from pathlib import Path
import os
import sys

from src.external.lafan1 import extract, utils
from src.models.silk import create_silk_model
from src.data_utils.features import extract_root_space_features_numpy
from src.evaluation.reconstruction import reconstruct_from_model_output


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(checkpoint_path, model):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def write_bvh(filename, anim, frametime=0.033333):
    """
    Write animation data to BVH file.

    Args:
        filename: Output BVH filename
        anim: Anim object with quats, pos, offsets, parents, bones
        frametime: Frame time in seconds (default: 1/30 = 0.033333)
    """
    with open(filename, 'w') as f:
        # Write hierarchy
        f.write("HIERARCHY\n")

        # Track which joints have been written
        written = [False] * len(anim.parents)

        def write_joint(joint_idx, indent=0):
            """Recursively write joint hierarchy."""
            indent_str = "  " * indent
            joint_name = anim.bones[joint_idx]
            offset = anim.offsets[joint_idx]

            # Determine joint type
            if joint_idx == 0:
                f.write(f"{indent_str}ROOT {joint_name}\n")
            else:
                f.write(f"{indent_str}JOINT {joint_name}\n")

            f.write(f"{indent_str}{{\n")
            f.write(f"{indent_str}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

            # Write channels (root has position + rotation, others just rotation)
            if joint_idx == 0:
                f.write(f"{indent_str}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
            else:
                f.write(f"{indent_str}  CHANNELS 3 Zrotation Yrotation Xrotation\n")

            written[joint_idx] = True

            # Find and write children
            children = [i for i, p in enumerate(anim.parents) if p == joint_idx and not written[i]]
            for child_idx in children:
                write_joint(child_idx, indent + 1)

            # Write end site if no children (leaf joint)
            if len(children) == 0 and joint_idx != 0:
                f.write(f"{indent_str}  End Site\n")
                f.write(f"{indent_str}  {{\n")
                f.write(f"{indent_str}    OFFSET 0.0 0.0 0.0\n")
                f.write(f"{indent_str}  }}\n")

            f.write(f"{indent_str}}}\n")

        # Write from root
        write_joint(0)

        # Write motion data
        f.write("MOTION\n")
        f.write(f"Frames: {len(anim.pos)}\n")
        f.write(f"Frame Time: {frametime:.6f}\n")

        # Convert quaternions to euler angles
        eulers = utils.quat_to_euler(anim.quats, order='zyx')
        eulers = np.degrees(eulers)  # Convert to degrees

        # Write each frame
        for frame_idx in range(len(anim.pos)):
            frame_data = []

            # Root position
            root_pos = anim.pos[frame_idx, 0]
            frame_data.extend([root_pos[0], root_pos[1], root_pos[2]])

            # All joint rotations (including root)
            for joint_idx in range(len(anim.parents)):
                euler = eulers[frame_idx, joint_idx]
                # BVH format: "Zrotation Yrotation Xrotation"
                # euler is [z, y, x], so write in that order
                frame_data.extend([euler[0], euler[1], euler[2]])

            # Write frame line
            f.write(" ".join([f"{x:.6f}" for x in frame_data]) + "\n")


def quat_to_euler(quats, order='zyx'):
    """
    Convert quaternions to euler angles.

    Implements the exact inverse of LAFAN1's euler_to_quat function.
    For 'zyx' order: q = qz * qy * qx where each q is a rotation around that axis.

    Args:
        quats: (T, J, 4) quaternions in [w, x, y, z] format
        order: rotation order (default: 'zyx')

    Returns:
        eulers: (T, J, 3) euler angles in radians
    """
    T, J, _ = quats.shape
    eulers = np.zeros((T, J, 3))

    for t in range(T):
        for j in range(J):
            q = quats[t, j]
            w, x, y, z = q[0], q[1], q[2], q[3]

            if order == 'zyx':
                # ZYX Euler angles (intrinsic rotations: Z, then Y, then X)
                # This matches LAFAN1's quat_mul(qz, quat_mul(qy, qx))

                # Extract Z rotation (yaw)
                euler_z = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

                # Extract Y rotation (pitch)
                sin_y = 2*(w*y - x*z)
                sin_y = np.clip(sin_y, -1.0, 1.0)
                euler_y = np.arcsin(sin_y)

                # Extract X rotation (roll)
                euler_x = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

                eulers[t, j] = [euler_z, euler_y, euler_x]
            else:
                # For other orders, could implement similarly or raise error
                raise NotImplementedError(f"Order '{order}' not implemented. Only 'zyx' is supported.")

    return eulers


# Monkey-patch the utils module
utils.quat_to_euler = quat_to_euler


def infer_bvh(
    bvh_path,
    start_context,
    start_predict,
    target,
    checkpoint_path,
    config_path,
    output_path,
    device='cpu'
):
    """
    Run SILK inference on a BVH file.

    Args:
        bvh_path: Path to input BVH file
        start_context: Frame index where context starts (e.g., 0)
        start_predict: Frame index where prediction/zero-filling starts (e.g., 10)
        target: Frame index of target keyframe (e.g., 40)
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_path: Path to output BVH file
        device: Device to run inference on
    """
    print("=" * 70)
    print("SILK INFERENCE")
    print("=" * 70)

    # Load config
    config = load_config(config_path)
    print(f"\nConfig: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input BVH: {bvh_path}")
    print(f"Output BVH: {output_path}")

    # Validate frame indices
    context_frames = start_predict - start_context
    transition_length = target - start_predict
    total_frames = target - start_context + 1

    print(f"\nFrame ranges:")
    print(f"  Context: frames {start_context} to {start_predict-1} ({context_frames} frames)")
    print(f"  Transition: frames {start_predict} to {target-1} ({transition_length} frames)")
    print(f"  Target: frame {target}")
    print(f"  Total: {total_frames} frames")

    if context_frames < 1:
        raise ValueError("Must have at least 1 context frame")
    if transition_length < 1:
        raise ValueError("Must have at least 1 transition frame")

    # Load BVH file
    print("\n" + "=" * 70)
    print("Loading BVH File")
    print("=" * 70)

    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    anim = extract.read_bvh(bvh_path)
    print(f"Loaded {anim.quats.shape[0]} frames with {anim.quats.shape[1]} joints")

    # Validate frame range
    if target >= anim.quats.shape[0]:
        raise ValueError(f"Target frame {target} exceeds BVH length {anim.quats.shape[0]}")

    # Extract sequence (MUST COPY to avoid modifying original)
    local_quats = anim.quats[start_context:target+1].copy()  # (T, J, 4)
    local_pos = anim.pos[start_context:target+1].copy()      # (T, J, 3)

    print(f"Extracted sequence shape: quats={local_quats.shape}, pos={local_pos.shape}")

    # ===== CRITICAL PREPROCESSING (matching LAFANDataset) =====
    print("\n" + "=" * 70)
    print("Applying Dataset Preprocessing")
    print("=" * 70)

    # Step 1: Compute global positions for centering
    global_quats_temp, global_pos_temp = utils.quat_fk(
        local_quats,
        local_pos,
        anim.parents
    )

    # Step 2: Center sequence around XZ = 0
    # This prevents root positions from being at absurd coordinates
    root_xz_mean = np.mean(global_pos_temp[:, 0, [0, 2]], axis=0, keepdims=True)
    local_pos[:, 0, 0] = local_pos[:, 0, 0] - root_xz_mean[0, 0]
    local_pos[:, 0, 2] = local_pos[:, 0, 2] - root_xz_mean[0, 1]
    print(f"Centered root XZ: offset = [{root_xz_mean[0, 0]:.2f}, {root_xz_mean[0, 1]:.2f}]")

    # Step 3: Unify facing direction at context frame
    # This makes the model rotation-invariant
    local_pos_batch = local_pos[np.newaxis, ...]  # Add batch dim
    local_quats_batch = local_quats[np.newaxis, ...]

    local_pos_batch, local_quats_batch = utils.rotate_at_frame(
        local_pos_batch,
        local_quats_batch,
        anim.parents,
        n_past=context_frames
    )

    local_pos = local_pos_batch[0]  # Remove batch dim
    local_quats = local_quats_batch[0]
    print(f"Rotated to unify facing at context frame {context_frames-1}")
    # ===== END PREPROCESSING =====

    # Convert to global coordinates (Forward Kinematics)
    print("\n" + "=" * 70)
    print("Computing Global Coordinates (FK)")
    print("=" * 70)

    global_quats, global_pos = utils.quat_fk(local_quats, local_pos, anim.parents)
    print(f"Global quats: {global_quats.shape}, Global pos: {global_pos.shape}")

    # Extract features
    print("\n" + "=" * 70)
    print("Extracting Root-Space Features")
    print("=" * 70)

    # Input features (with velocity)
    input_features = extract_root_space_features_numpy(
        global_quats, global_pos, include_velocity=True
    )

    # Output features (without velocity)
    output_features = extract_root_space_features_numpy(
        global_quats, global_pos, include_velocity=False
    )

    print(f"Input features (din): {input_features.shape}")
    print(f"Output features (dout): {output_features.shape}")

    # Load training statistics
    print("\n" + "=" * 70)
    print("Loading Training Statistics")
    print("=" * 70)

    stats_path = Path(checkpoint_path).parent / 'train_stats.pkl'
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Training stats not found: {stats_path}")

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print(f"Loaded stats from: {stats_path}")

    # Normalize features
    input_mean = stats['input_mean']
    input_std = stats['input_std']
    output_mean = stats['output_mean']
    output_std = stats['output_std']

    input_normalized = (input_features - input_mean) / input_std
    output_normalized = (output_features - output_mean) / output_std

    # Zero-fill transition frames in input
    print("\n" + "=" * 70)
    print("Creating Zero-Filled Input")
    print("=" * 70)

    trans_start = start_predict - start_context
    trans_end = target - start_context

    print(f"Zero-filling frames {trans_start} to {trans_end-1} (transition)")
    input_normalized[trans_start:trans_end] = 0.0

    # Convert to torch tensors
    input_tensor = torch.from_numpy(input_normalized).float().unsqueeze(0).to(device)  # (1, T, din)
    output_tensor = torch.from_numpy(output_normalized).float().unsqueeze(0).to(device)  # (1, T, dout)

    # Load model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    model = create_silk_model(num_joints=config['data']['num_joints'])
    model = load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Run inference
    print("\n" + "=" * 70)
    print("Running Inference")
    print("=" * 70)

    with torch.no_grad():
        pred_output = model(input_tensor)  # (1, T, dout)

    print(f"Prediction shape: {pred_output.shape}")

    # Combine predictions with ground truth for context and target
    print("\n" + "=" * 70)
    print("Combining Predictions with Ground Truth")
    print("=" * 70)

    output_combined = output_tensor.clone()
    output_combined[:, trans_start:trans_end, :] = pred_output[:, trans_start:trans_end, :]

    print(f"Using ground truth for context frames 0-{trans_start-1}")
    print(f"Using predictions for transition frames {trans_start}-{trans_end-1}")
    print(f"Using ground truth for target frame {trans_end}")

    # Reconstruct global coordinates
    print("\n" + "=" * 70)
    print("Reconstructing Global Coordinates")
    print("=" * 70)

    pred_global_quats, pred_global_pos = reconstruct_from_model_output(
        output_combined,
        stats,
        num_joints=config['data']['num_joints']
    )

    # Remove batch dimension
    pred_global_quats = pred_global_quats[0]  # (T, J, 4)
    pred_global_pos = pred_global_pos[0]      # (T, J, 3)

    print(f"Reconstructed global quats: {pred_global_quats.shape}")
    print(f"Reconstructed global pos: {pred_global_pos.shape}")

    # Convert back to local coordinates (Inverse Kinematics)
    print("\n" + "=" * 70)
    print("Converting to Local Coordinates (IK)")
    print("=" * 70)

    pred_local_quats, pred_local_pos = utils.quat_ik(
        pred_global_quats,
        pred_global_pos,
        anim.parents
    )

    print(f"Local quats: {pred_local_quats.shape}")
    print(f"Local pos: {pred_local_pos.shape}")

    # ===== INVERSE PREPROCESSING (undo the transformations) =====
    print("\n" + "=" * 70)
    print("Undoing Preprocessing Transformations")
    print("=" * 70)

    # Step 1: Undo rotation normalization
    # Rotate back from normalized facing direction
    pred_local_pos_batch = pred_local_pos[np.newaxis, ...]
    pred_local_quats_batch = pred_local_quats[np.newaxis, ...]

    # Get the inverse rotation (from context frame)
    # We need to rotate back by the opposite of what we did before
    # rotate_at_frame rotated TO a canonical orientation, we need to rotate BACK
    # Unfortunately, utils.rotate_at_frame doesn't provide the inverse directly
    # So we need to compute the rotation that was applied and invert it

    # Actually, for simplicity, let's just apply the rotation from the ORIGINAL data
    # at the context frame to the predicted data
    # Get original orientation at context frame (before preprocessing)
    original_local_quats = anim.quats[start_context:target+1].copy()
    original_local_pos = anim.pos[start_context:target+1].copy()

    # Compute the global quaternion at context frame from ORIGINAL data
    global_quats_orig, global_pos_orig = utils.quat_fk(
        original_local_quats,
        original_local_pos,
        anim.parents
    )

    # Get the root orientation at context frame from original
    key_glob_Q_orig = global_quats_orig[context_frames-1:context_frames, 0:1, :]  # (1, 1, 4)

    # Compute forward vector from original orientation
    forward_orig = np.array([1, 0, 1])[np.newaxis, np.newaxis, :] \
                   * utils.quat_mul_vec(key_glob_Q_orig, np.array([0, 1, 0])[np.newaxis, np.newaxis, :])
    forward_orig = utils.normalize(forward_orig)
    yrot_orig = utils.quat_normalize(utils.quat_between(np.array([1, 0, 0]), forward_orig))

    # Apply this rotation to the predicted data
    pred_global_quats_final, pred_global_pos_final = utils.quat_fk(
        pred_local_quats,
        pred_local_pos,
        anim.parents
    )

    # Rotate by the original orientation
    new_glob_Q = utils.quat_mul(yrot_orig, pred_global_quats_final)
    new_glob_X = utils.quat_mul_vec(yrot_orig, pred_global_pos_final)

    # Convert back to local
    pred_local_quats, pred_local_pos = utils.quat_ik(new_glob_Q, new_glob_X, anim.parents)

    print(f"Rotated back to original facing direction")

    # Step 2: Undo centering (add back the XZ offset)
    pred_local_pos[:, 0, 0] = pred_local_pos[:, 0, 0] + root_xz_mean[0, 0]
    pred_local_pos[:, 0, 2] = pred_local_pos[:, 0, 2] + root_xz_mean[0, 1]
    print(f"Uncentered root XZ: offset = [{root_xz_mean[0, 0]:.2f}, {root_xz_mean[0, 1]:.2f}]")
    # ===== END INVERSE PREPROCESSING =====

    # Create output animation object
    output_anim = extract.Anim(
        quats=pred_local_quats,
        pos=pred_local_pos,
        offsets=anim.offsets,
        parents=anim.parents,
        bones=anim.bones
    )

    # Write to BVH file
    print("\n" + "=" * 70)
    print("Writing Output BVH")
    print("=" * 70)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    write_bvh(output_path, output_anim, frametime=0.033333)

    print(f"✓ Output written to: {output_path}")
    print(f"  Frames: {len(pred_local_quats)}")
    print(f"  Joints: {len(anim.parents)}")

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Run SILK inference on a BVH file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate transition from frame 0-10 to frame 40
  python infer.py --bvh input.bvh --start-context 0 --start-predict 10 --target 40 \\
                  --checkpoint checkpoints/best_model.pth --output output.bvh

  # 5-frame transition
  python infer.py --bvh input.bvh --start-context 0 --start-predict 10 --target 15 \\
                  --checkpoint checkpoints/best_model.pth --output output.bvh
        """
    )

    parser.add_argument('--bvh', type=str, required=True,
                       help='Path to input BVH file')
    parser.add_argument('--start-context', type=int, required=True,
                       help='Frame index where context starts (inclusive)')
    parser.add_argument('--start-predict', type=int, required=True,
                       help='Frame index where prediction/zero-filling starts (inclusive)')
    parser.add_argument('--target', type=int, required=True,
                       help='Frame index of target keyframe (inclusive)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, default='configs/silk_default.yaml',
                       help='Path to config file (default: configs/silk_default.yaml)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output BVH file')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run inference on (default: cpu)')

    args = parser.parse_args()

    try:
        infer_bvh(
            bvh_path=args.bvh,
            start_context=args.start_context,
            start_predict=args.start_predict,
            target=args.target,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            output_path=args.output,
            device=args.device
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
