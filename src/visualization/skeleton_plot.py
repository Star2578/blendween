"""
3D Skeleton Visualization

Tools for plotting LAFAN1 skeleton poses and motion sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import sys
import os

# Add LAFAN1 utilities
from src.external.lafan1 import extract

def create_skeleton_lines(parents):
    """
    Create line segments connecting joints based on skeleton hierarchy.

    Args:
        parents: Array of parent indices defining skeleton hierarchy

    Returns:
        lines: List of (parent_idx, child_idx) tuples for drawing bones
    """
    lines = []
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx != -1:  # Skip root (has no parent)
            lines.append((parent_idx, child_idx))
    return lines


def plot_skeleton_3d(positions, parents=None, ax=None, title="Skeleton Pose",
                     color='b', alpha=1.0, label=None):
    """
    Plot a 3D skeleton at a single timestep.

    Args:
        positions: Joint positions array of shape (num_joints, 3) or (1, num_joints, 3)
        parents: Parent indices (if None, loads default LAFAN1 skeleton)
        ax: Matplotlib 3D axis (creates new if None)
        title: Plot title
        color: Color for skeleton bones
        alpha: Transparency (0-1)
        label: Label for legend

    Returns:
        ax: Matplotlib 3D axis
    """
    # Handle batch dimension
    if positions.ndim == 3:
        positions = positions[0]  # Take first in batch

    # Load default skeleton if not provided
    if parents is None:
        # Load a sample BVH to get skeleton structure
        sample_bvh = 'dataset/ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh'
        if os.path.exists(sample_bvh):
            anim = extract.read_bvh(sample_bvh)
            parents = anim.parents
        else:
            raise ValueError("Cannot load skeleton structure. Provide 'parents' parameter.")

    # Create axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Get skeleton lines
    lines = create_skeleton_lines(parents)

    # Plot joints as points
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=color, marker='o', s=50, alpha=alpha, label=label)

    # Plot bones as lines
    for parent_idx, child_idx in lines:
        parent_pos = positions[parent_idx]
        child_pos = positions[child_idx]
        ax.plot([parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                [parent_pos[2], child_pos[2]],
                c=color, linewidth=2, alpha=alpha)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Good viewing angle
    ax.view_init(elev=20, azim=45)

    if label:
        ax.legend()

    return ax


def plot_motion_sequence(positions, parents=None, fps=30, output_path=None,
                         title="Motion Sequence", interval=33):
    """
    Create an animation of a motion sequence.

    Args:
        positions: Joint positions array of shape (timesteps, num_joints, 3)
        parents: Parent indices (if None, loads default LAFAN1 skeleton)
        fps: Frames per second for display
        output_path: If provided, save animation to this path (e.g., 'output.mp4' or 'output.gif')
        title: Animation title
        interval: Milliseconds between frames (default: 33ms ≈ 30fps)

    Returns:
        anim: FuncAnimation object
    """
    # Load default skeleton if not provided
    if parents is None:
        sample_bvh = 'dataset/ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh'
        if os.path.exists(sample_bvh):
            anim_data = extract.read_bvh(sample_bvh)
            parents = anim_data.parents
        else:
            raise ValueError("Cannot load skeleton structure. Provide 'parents' parameter.")

    timesteps = positions.shape[0]
    lines_data = create_skeleton_lines(parents)

    # Set up the figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Compute global bounds for all frames
    all_positions = positions.reshape(-1, 3)
    max_range = np.array([
        all_positions[:, 0].max() - all_positions[:, 0].min(),
        all_positions[:, 1].max() - all_positions[:, 1].min(),
        all_positions[:, 2].max() - all_positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 0].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5

    def init():
        ax.clear()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title} - Frame {frame}/{timesteps}')
        ax.view_init(elev=20, azim=45)

        pos = positions[frame]

        # Plot joints
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                  c='b', marker='o', s=50)

        # Plot bones
        for parent_idx, child_idx in lines_data:
            parent_pos = pos[parent_idx]
            child_pos = pos[child_idx]
            ax.plot([parent_pos[0], child_pos[0]],
                   [parent_pos[1], child_pos[1]],
                   [parent_pos[2], child_pos[2]],
                   c='b', linewidth=2)

        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=timesteps,
                        interval=interval, blit=True)

    # Save if output path provided
    if output_path:
        print(f"Saving animation to {output_path}...")
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        elif output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=fps)
        else:
            raise ValueError("output_path must end with .gif or .mp4")
        print(f"Animation saved!")

    return anim


def plot_comparison(gt_positions, pred_positions, parents=None, frame_idx=0,
                   output_path=None, title="Ground Truth vs Prediction"):
    """
    Plot ground truth and predicted skeletons side by side.

    Args:
        gt_positions: Ground truth positions (timesteps, num_joints, 3)
        pred_positions: Predicted positions (timesteps, num_joints, 3)
        parents: Parent indices
        frame_idx: Which frame to visualize
        output_path: If provided, save figure to this path
        title: Figure title

    Returns:
        fig: Matplotlib figure
    """
    # Load default skeleton if not provided
    if parents is None:
        sample_bvh = 'dataset/ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh'
        if os.path.exists(sample_bvh):
            anim = extract.read_bvh(sample_bvh)
            parents = anim.parents
        else:
            raise ValueError("Cannot load skeleton structure. Provide 'parents' parameter.")

    fig = plt.figure(figsize=(16, 8))

    # Plot ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    plot_skeleton_3d(gt_positions[frame_idx], parents, ax=ax1,
                    title="Ground Truth", color='g')

    # Plot prediction
    ax2 = fig.add_subplot(122, projection='3d')
    plot_skeleton_3d(pred_positions[frame_idx], parents, ax=ax2,
                    title="Prediction", color='r')

    plt.suptitle(f'{title} - Frame {frame_idx}')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")

    return fig


def plot_overlay(gt_positions, pred_positions, parents=None, frame_idx=0,
                output_path=None, title="Overlay Comparison"):
    """
    Plot ground truth and predicted skeletons overlaid on same axis.

    Args:
        gt_positions: Ground truth positions (timesteps, num_joints, 3)
        pred_positions: Predicted positions (timesteps, num_joints, 3)
        parents: Parent indices
        frame_idx: Which frame to visualize
        output_path: If provided, save figure to this path
        title: Figure title

    Returns:
        fig: Matplotlib figure
    """
    # Load default skeleton if not provided
    if parents is None:
        sample_bvh = 'dataset/ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh'
        if os.path.exists(sample_bvh):
            anim = extract.read_bvh(sample_bvh)
            parents = anim.parents
        else:
            raise ValueError("Cannot load skeleton structure. Provide 'parents' parameter.")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot both skeletons on same axis
    plot_skeleton_3d(gt_positions[frame_idx], parents, ax=ax,
                    color='g', alpha=0.7, label='Ground Truth')
    plot_skeleton_3d(pred_positions[frame_idx], parents, ax=ax,
                    color='r', alpha=0.7, label='Prediction')

    ax.set_title(f'{title} - Frame {frame_idx}')
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved overlay to {output_path}")

    return fig


if __name__ == "__main__":
    # Test visualization with sample data
    print("Testing skeleton visualization...")

    # Load a sample BVH file
    sample_bvh = 'dataset/ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh'

    if os.path.exists(sample_bvh):
        print(f"Loading {sample_bvh}...")
        anim = extract.read_bvh(sample_bvh)

        # Add path for FK
        from src.external.lafan1 import utils

        # Compute global positions via FK
        global_quats, global_pos = utils.quat_fk(
            anim.quats[:50],  # First 50 frames
            anim.pos[:50],
            anim.parents
        )

        print(f"Loaded {global_pos.shape[0]} frames with {global_pos.shape[1]} joints")

        # Test single frame plot
        print("\n1. Plotting single frame...")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plot_skeleton_3d(global_pos[0], anim.parents, ax=ax, title="Frame 0")
        plt.savefig('outputs/test_skeleton_frame.png', dpi=150, bbox_inches='tight')
        print("   Saved to outputs/test_skeleton_frame.png")
        plt.close()

        # Test comparison plot
        print("\n2. Plotting comparison...")
        fig = plot_comparison(
            global_pos,
            global_pos + np.random.randn(*global_pos.shape) * 0.05,  # Add noise for demo
            anim.parents,
            frame_idx=10,
            output_path='outputs/test_comparison.png'
        )
        plt.close()

        # Test overlay
        print("\n3. Plotting overlay...")
        fig = plot_overlay(
            global_pos,
            global_pos + np.random.randn(*global_pos.shape) * 0.05,
            anim.parents,
            frame_idx=10,
            output_path='outputs/test_overlay.png'
        )
        plt.close()

        print("\nVisualization tools working correctly!")
        print("Check outputs/ directory for generated plots")
    else:
        print(f"Sample BVH not found at {sample_bvh}")
        print("Please run with a valid BVH file path")
