"""
Visualize SILK Predictions with Full Animation

Creates animations showing ground truth vs predicted motion sequences.

Usage:
    python visualize_predictions.py \
        --checkpoint checkpoints/best_model.pth \
        --config configs/silk_default.yaml \
        --sequence-idx 0 \
        --transition-length 30
"""

import sys
import os

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import pickle

from src.models.silk import create_silk_model
from src.data.lafan_dataset import LAFANDataset
from src.evaluation.reconstruction import reconstruct_from_model_output
from src.visualization import create_skeleton_lines


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


def create_evaluation_sequence(sequence, context_frames, transition_length, device):
    """Create evaluation sequence with zero-filling."""
    input_features = sequence['input']
    output_features = sequence['output']

    seq_len = context_frames + transition_length + 1

    if seq_len > len(input_features):
        raise ValueError(f"Sequence too short: need {seq_len} frames, have {len(input_features)}")

    input_seq = input_features[:seq_len].clone()
    target_seq = output_features[:seq_len].clone()

    # Zero-fill transition frames
    input_seq[context_frames:seq_len-1] = 0.0

    # Add batch dimension
    input_seq = input_seq.unsqueeze(0).to(device)
    target_seq = target_seq.unsqueeze(0).to(device)

    return input_seq, target_seq


def create_side_by_side_animation(gt_pos, pred_pos, parents, output_path,
                                   context_frames=10, fps=30):
    """
    Create side-by-side animation of GT vs prediction.

    Args:
        gt_pos: Ground truth positions (timesteps, joints, 3)
        pred_pos: Predicted positions (timesteps, joints, 3)
        parents: Parent indices
        output_path: Where to save animation
        context_frames: Number of context frames (highlighted differently)
        fps: Frames per second
    """
    timesteps = gt_pos.shape[0]
    lines_data = create_skeleton_lines(parents)

    # Set up figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Compute global bounds
    all_pos = np.concatenate([gt_pos, pred_pos], axis=0)
    max_range = np.array([
        all_pos[:, :, 0].max() - all_pos[:, :, 0].min(),
        all_pos[:, :, 1].max() - all_pos[:, :, 1].min(),
        all_pos[:, :, 2].max() - all_pos[:, :, 2].min()
    ]).max() / 2.0

    mid_x = (all_pos[:, :, 0].max() + all_pos[:, :, 0].min()) * 0.5
    mid_y = (all_pos[:, :, 1].max() + all_pos[:, :, 1].min()) * 0.5
    mid_z = (all_pos[:, :, 2].max() + all_pos[:, :, 2].min()) * 0.5

    def update(frame):
        # Clear both axes
        ax1.clear()
        ax2.clear()

        # Set consistent limits
        for ax in [ax1, ax2]:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20, azim=45)

        # Determine if this is a context or transition frame
        is_context = frame < context_frames
        is_target = frame == timesteps - 1

        if is_context:
            frame_type = "Context"
            color_gt = 'blue'
            color_pred = 'blue'
        elif is_target:
            frame_type = "Target Keyframe"
            color_gt = 'purple'
            color_pred = 'purple'
        else:
            frame_type = "Transition (Predicted)"
            color_gt = 'green'
            color_pred = 'red'

        # Plot ground truth
        gt = gt_pos[frame]
        ax1.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c=color_gt, marker='o', s=50)
        for parent_idx, child_idx in lines_data:
            ax1.plot([gt[parent_idx, 0], gt[child_idx, 0]],
                    [gt[parent_idx, 1], gt[child_idx, 1]],
                    [gt[parent_idx, 2], gt[child_idx, 2]],
                    c=color_gt, linewidth=2)
        ax1.set_title(f'Ground Truth\n{frame_type}', fontsize=12, fontweight='bold')

        # Plot prediction
        pred = pred_pos[frame]
        ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=color_pred, marker='o', s=50)
        for parent_idx, child_idx in lines_data:
            ax2.plot([pred[parent_idx, 0], pred[child_idx, 0]],
                    [pred[parent_idx, 1], pred[child_idx, 1]],
                    [pred[parent_idx, 2], pred[child_idx, 2]],
                    c=color_pred, linewidth=2)
        ax2.set_title(f'SILK Prediction\n{frame_type}', fontsize=12, fontweight='bold')

        # Overall title
        fig.suptitle(f'Frame {frame + 1}/{timesteps}', fontsize=14, fontweight='bold')

        return []

    anim = FuncAnimation(fig, update, frames=timesteps, interval=int(1000/fps), blit=True)

    print(f"Saving animation to {output_path}...")
    if output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=fps)
    elif output_path.endswith('.mp4'):
        anim.save(output_path, writer='ffmpeg', fps=fps)
    print(f"Animation saved!")

    return anim


def create_overlay_animation(gt_pos, pred_pos, parents, output_path,
                             context_frames=10, fps=30):
    """
    Create overlay animation with GT and prediction on same plot.

    Args:
        gt_pos: Ground truth positions (timesteps, joints, 3)
        pred_pos: Predicted positions (timesteps, joints, 3)
        parents: Parent indices
        output_path: Where to save animation
        context_frames: Number of context frames
        fps: Frames per second
    """
    timesteps = gt_pos.shape[0]
    lines_data = create_skeleton_lines(parents)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Compute global bounds
    all_pos = np.concatenate([gt_pos, pred_pos], axis=0)
    max_range = np.array([
        all_pos[:, :, 0].max() - all_pos[:, :, 0].min(),
        all_pos[:, :, 1].max() - all_pos[:, :, 1].min(),
        all_pos[:, :, 2].max() - all_pos[:, :, 2].min()
    ]).max() / 2.0

    mid_x = (all_pos[:, :, 0].max() + all_pos[:, :, 0].min()) * 0.5
    mid_y = (all_pos[:, :, 1].max() + all_pos[:, :, 1].min()) * 0.5
    mid_z = (all_pos[:, :, 2].max() + all_pos[:, :, 2].min()) * 0.5

    def update(frame):
        ax.clear()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)

        # Determine frame type
        is_context = frame < context_frames
        is_target = frame == timesteps - 1

        if is_context:
            frame_type = "Context Frame"
        elif is_target:
            frame_type = "Target Keyframe"
        else:
            frame_type = "Transition Frame"

        # Plot both skeletons
        gt = gt_pos[frame]
        pred = pred_pos[frame]

        # Ground truth (green, semi-transparent)
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='green', marker='o', s=50, alpha=0.6, label='Ground Truth')
        for parent_idx, child_idx in lines_data:
            ax.plot([gt[parent_idx, 0], gt[child_idx, 0]],
                   [gt[parent_idx, 1], gt[child_idx, 1]],
                   [gt[parent_idx, 2], gt[child_idx, 2]],
                   c='green', linewidth=2, alpha=0.6)

        # Prediction (red, semi-transparent)
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', marker='o', s=50, alpha=0.6, label='Prediction')
        for parent_idx, child_idx in lines_data:
            ax.plot([pred[parent_idx, 0], pred[child_idx, 0]],
                   [pred[parent_idx, 1], pred[child_idx, 1]],
                   [pred[parent_idx, 2], pred[child_idx, 2]],
                   c='red', linewidth=2, alpha=0.6)

        ax.set_title(f'{frame_type} - Frame {frame + 1}/{timesteps}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        return []

    anim = FuncAnimation(fig, update, frames=timesteps, interval=int(1000/fps), blit=True)

    print(f"Saving overlay animation to {output_path}...")
    if output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=fps)
    elif output_path.endswith('.mp4'):
        anim.save(output_path, writer='ffmpeg', fps=fps)
    print(f"Overlay animation saved!")

    return anim


def main(args):
    """Main function."""
    print("=" * 70)
    print("SILK PREDICTION VISUALIZATION")
    print("=" * 70)

    # Load configuration
    config = load_config(args.config)
    print(f"Configuration: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load training statistics
    stats_path = Path(args.checkpoint).parent / 'train_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Load test dataset
    print("\n" + "=" * 70)
    print("Loading Test Dataset")
    print("=" * 70)

    test_dataset = LAFANDataset(
        bvh_folder=config['data']['dataset_path'],
        actors=config['data']['test_actors'],
        window_size=65,
        offset=40,
        context_frames=config['data']['context_frames'],
        train_stats=stats,
        cache_path='data/cache/test_viz.pkl'
    )

    # Load skeleton structure
    from src.external.lafan1 import extract
    sample_bvh = os.path.join(config['data']['dataset_path'], 'aiming2_subject5.bvh')
    if not os.path.exists(sample_bvh):
        # Try any subject 5 file
        import glob
        subject5_files = glob.glob(os.path.join(config['data']['dataset_path'], '*subject5.bvh'))
        if subject5_files:
            sample_bvh = subject5_files[0]
    anim = extract.read_bvh(sample_bvh)
    parents = anim.parents

    # Create model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    model = create_silk_model(num_joints=config['data']['num_joints'])
    model = load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    # Get sequence
    print("\n" + "=" * 70)
    print("Generating Predictions")
    print("=" * 70)

    sequence = test_dataset[args.sequence_idx]
    context_frames = config['data']['context_frames']
    transition_length = args.transition_length

    print(f"Sequence index: {args.sequence_idx}")
    print(f"Context frames: {context_frames}")
    print(f"Transition length: {transition_length}")
    print(f"Total frames: {context_frames + transition_length + 1}")

    # Create input and get prediction
    try:
        input_seq, target_seq = create_evaluation_sequence(
            sequence, context_frames, transition_length, device
        )
    except ValueError as e:
        print(f"\nError: {e}")
        return

    # DEBUG: Print root position frame by frame
    print("\n" + "=" * 70)
    print("DEBUG: Root Position (target_seq) Frame by Frame")
    print("=" * 70)
    print("Format: root_x, root_z, root_cos, root_sin (first 4 dims of output)")
    print("-" * 70)

    target_seq_cpu = target_seq.cpu().numpy()[0]  # Remove batch dim
    for i in range(len(target_seq_cpu)):
        root_pos = target_seq_cpu[i, :4]  # First 4 dimensions are root
        print(f"Frame {i:2d}: x={root_pos[0]:8.4f}, z={root_pos[1]:8.4f}, "
              f"cos={root_pos[2]:7.4f}, sin={root_pos[3]:7.4f}")
    print("=" * 70)

    with torch.no_grad():
        pred_seq = model(input_seq)

    # Reconstruct ground truth for full sequence
    gt_quats, gt_pos = reconstruct_from_model_output(
        target_seq,
        stats,
        num_joints=22
    )

    # For prediction, only use model output for TRANSITION frames
    # Use ground truth for context and target frames
    pred_full = target_seq.clone()
    trans_start = context_frames
    trans_end = context_frames + transition_length
    pred_full[:, trans_start:trans_end, :] = pred_seq[:, trans_start:trans_end, :]

    pred_quats, pred_pos = reconstruct_from_model_output(
        pred_full,
        stats,
        num_joints=22
    )

    # Remove batch dimension
    pred_pos = pred_pos[0]  # (seq_len, 22, 3)
    gt_pos = gt_pos[0]

    print(f"\nReconstructed positions shape: {pred_pos.shape}")

    # DIAGNOSTIC: Check if context and target frames match between GT and pred
    print(f"\n" + "=" * 70)
    print("DIAGNOSTIC: Checking context and target frame alignment")
    print("=" * 70)

    # Check first context frame (should be identical)
    context_diff = np.abs(pred_pos[0] - gt_pos[0]).mean()
    print(f"Context frame 0 difference (GT vs Pred): {context_diff:.6f} cm")
    if context_diff < 0.01:
        print("  Context frames match (using GT)")
    else:
        print(f"  ❌ Context frames DON'T match! (diff = {context_diff:.2f} cm)")

    # Check target frame (should be identical)
    target_idx = len(pred_pos) - 1
    target_diff = np.abs(pred_pos[target_idx] - gt_pos[target_idx]).mean()
    print(f"\nTarget frame {target_idx} difference (GT vs Pred): {target_diff:.6f} cm")
    if target_diff < 0.01:
        print("  Target frames match (using GT)")
    else:
        print(f"  ❌ Target frames DON'T match! (diff = {target_diff:.2f} cm)")
        print(f"\n  This means pred_full is using model prediction for target frame!")
        print(f"  trans_start={trans_start}, trans_end={trans_end}, target_idx={target_idx}")
        print(f"  Check if trans_end is correct (should be < target_idx)")

    # Check a transition frame (should be different)
    trans_idx = context_frames + 5
    trans_diff = np.abs(pred_pos[trans_idx] - gt_pos[trans_idx]).mean()
    print(f"\nTransition frame {trans_idx} difference (GT vs Pred): {trans_diff:.6f} cm")
    if trans_diff > 1.0:
        print(f"  Transition frames differ (using model prediction)")
    else:
        print(f"  ⚠ Transition frames are too similar (diff = {trans_diff:.2f} cm)")

    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create animations
    print("\n" + "=" * 70)
    print("Creating Animations")
    print("=" * 70)

    # 1. Side-by-side comparison
    print("\n1. Side-by-side comparison...")
    side_by_side_path = output_dir / f'comparison_seq{args.sequence_idx}_trans{transition_length}.gif'
    create_side_by_side_animation(
        gt_pos, pred_pos, parents,
        output_path=str(side_by_side_path),
        context_frames=context_frames,
        fps=args.fps
    )

    # 2. Overlay animation
    print("\n2. Overlay animation...")
    overlay_path = output_dir / f'overlay_seq{args.sequence_idx}_trans{transition_length}.gif'
    create_overlay_animation(
        gt_pos, pred_pos, parents,
        output_path=str(overlay_path),
        context_frames=context_frames,
        fps=args.fps
    )

    # 3. Individual animations
    if args.save_individual:
        print("\n3. Individual animations...")
        from src.visualization import plot_motion_sequence

        gt_path = output_dir / f'gt_seq{args.sequence_idx}_trans{transition_length}.gif'
        plot_motion_sequence(gt_pos, parents, fps=args.fps,
                           output_path=str(gt_path), title="Ground Truth")

        pred_path = output_dir / f'pred_seq{args.sequence_idx}_trans{transition_length}.gif'
        plot_motion_sequence(pred_pos, parents, fps=args.fps,
                           output_path=str(pred_path), title="SILK Prediction")

    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {side_by_side_path.name}")
    print(f"  - {overlay_path.name}")
    if args.save_individual:
        print(f"  - gt_seq{args.sequence_idx}_trans{transition_length}.gif")
        print(f"  - pred_seq{args.sequence_idx}_trans{transition_length}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize SILK predictions with animations')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/silk_default.yaml',
                       help='Path to config file')
    parser.add_argument('--sequence-idx', type=int, default=0,
                       help='Which test sequence to visualize (default: 0)')
    parser.add_argument('--transition-length', type=int, default=30,
                       help='Transition length to generate (default: 30)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for animation (default: 30)')
    parser.add_argument('--output', type=str, default='outputs/animations',
                       help='Output directory for animations')
    parser.add_argument('--save-individual', action='store_true',
                       help='Also save separate GT and prediction animations')

    args = parser.parse_args()
    main(args)
