#!/usr/bin/env python3
"""Extract a frame range from a BVH file."""
import sys
import argparse
from src.external.lafan1 import extract
from infer import write_bvh

def extract_frames(input_bvh, output_bvh, start, end):
    """Extract frames from start to end (inclusive)."""
    # Read BVH
    anim = extract.read_bvh(input_bvh)
    print(f"Loaded {anim.quats.shape[0]} frames")

    # Extract frames
    quats = anim.quats[start:end+1]
    pos = anim.pos[start:end+1]
    print(f"Extracted frames {start} to {end} ({len(quats)} frames)")

    # Create new animation
    anim_out = extract.Anim(
        quats=quats,
        pos=pos,
        offsets=anim.offsets,
        parents=anim.parents,
        bones=anim.bones
    )

    # Write
    write_bvh(output_bvh, anim_out)
    print(f"Written to: {output_bvh}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input BVH')
    parser.add_argument('--output', required=True, help='Output BVH')
    parser.add_argument('--start', type=int, required=True, help='Start frame')
    parser.add_argument('--end', type=int, required=True, help='End frame')
    args = parser.parse_args()

    extract_frames(args.input, args.output, args.start, args.end)
