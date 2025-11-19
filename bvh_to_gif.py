#!/usr/bin/env python3
"""
BVH to GIF Converter (Fast version using imageio)

Reads a BVH motion capture file and generates an animated GIF.

Usage:
    python bvh_to_gif.py <bvh_file> [output_gif] [--fps FPS] [--max-frames FRAMES]

Examples:
    python bvh_to_gif.py dataset/ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh
    python bvh_to_gif.py input.bvh output.gif --fps 30
    python bvh_to_gif.py input.bvh animation.gif --max-frames 100
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import io
from PIL import Image

try:
    import imageio
except ImportError:
    print("Error: imageio not found. Install with: pip install imageio")
    sys.exit(1)

from src.external.lafan1 import extract, utils


def create_skeleton_lines(parents):
    """Create line segments connecting joints based on skeleton hierarchy."""
    lines = []
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx != -1:
            lines.append((parent_idx, child_idx))
    return lines


def render_frame_to_image(positions, parents, lines, bounds, title="", frame_num=0, total_frames=0):
    """Render a single frame to an image array."""
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c='b', marker='o', s=50)

    # Plot bones
    for parent_idx, child_idx in lines:
        parent_pos = positions[parent_idx]
        child_pos = positions[child_idx]
        ax.plot([parent_pos[0], child_pos[0]],
               [parent_pos[1], child_pos[1]],
               [parent_pos[2], child_pos[2]],
               c='b', linewidth=2)

    # Set fixed bounds
    mid_x, mid_y, mid_z, max_range = bounds
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title} - Frame {frame_num}/{total_frames}')
    ax.view_init(elev=20, azim=45)

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)

    plt.close(fig)
    buf.close()

    return img_array


def bvh_to_gif(bvh_path, output_path=None, fps=30, max_frames=None):
    """
    Convert a BVH file to an animated GIF (fast version using imageio).

    Args:
        bvh_path: Path to input BVH file
        output_path: Path to output GIF (default: same name as BVH with .gif extension)
        fps: Frames per second for the animation (default: 30)
        max_frames: Maximum number of frames to include (default: all frames)

    Returns:
        output_path: Path where GIF was saved
    """
    # Check if BVH file exists
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.splitext(bvh_path)[0] + '.gif'

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading BVH file: {bvh_path}")
    anim = extract.read_bvh(bvh_path)

    # Extract motion data
    num_frames = anim.quats.shape[0]
    print(f"Loaded {num_frames} frames with {anim.quats.shape[1]} joints")

    # Limit frames if specified
    if max_frames is not None and num_frames > max_frames:
        print(f"Limiting to first {max_frames} frames")
        quats = anim.quats[:max_frames]
        pos = anim.pos[:max_frames]
    else:
        quats = anim.quats
        pos = anim.pos

    # Compute global joint positions via forward kinematics
    print("Computing global joint positions...")
    global_quats, global_pos = utils.quat_fk(quats, pos, anim.parents)

    # Create skeleton lines
    lines = create_skeleton_lines(anim.parents)

    # Compute global bounds for all frames
    print("Computing scene bounds...")
    all_positions = global_pos.reshape(-1, 3)
    max_range = np.array([
        all_positions[:, 0].max() - all_positions[:, 0].min(),
        all_positions[:, 1].max() - all_positions[:, 1].min(),
        all_positions[:, 2].max() - all_positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
    bounds = (mid_x, mid_y, mid_z, max_range)

    # Render frames with progress bar
    print(f"Rendering {global_pos.shape[0]} frames...")
    frames = []
    title = os.path.basename(bvh_path)

    for i in tqdm(range(global_pos.shape[0]), desc="Rendering"):
        img = render_frame_to_image(
            global_pos[i],
            anim.parents,
            lines,
            bounds,
            title=title,
            frame_num=i,
            total_frames=global_pos.shape[0]
        )
        frames.append(img)

    # Save as GIF using imageio (much faster than matplotlib)
    print(f"Saving GIF to {output_path}...")
    duration = 1000 / fps  # duration per frame in milliseconds
    imageio.mimsave(output_path, frames, duration=duration, loop=0)

    print(f"✓ GIF saved to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert BVH motion capture file to animated GIF (fast version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bvh_to_gif.py input.bvh
  python bvh_to_gif.py input.bvh output.gif
  python bvh_to_gif.py input.bvh --fps 30 --max-frames 100
        """
    )

    parser.add_argument(
        'bvh_file',
        help='Path to input BVH file'
    )

    parser.add_argument(
        'output_gif',
        nargs='?',
        default=None,
        help='Path to output GIF (default: same name as BVH with .gif extension)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for animation (default: 30)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to include (default: all frames)'
    )

    args = parser.parse_args()

    try:
        bvh_to_gif(
            args.bvh_file,
            args.output_gif,
            fps=args.fps,
            max_frames=args.max_frames
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
