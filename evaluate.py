"""
SILK Evaluation Script

Evaluates a trained SILK model on the LAFAN1 test set and computes
standard metrics (L2P, L2Q, NPSS) at different transition lengths.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --config configs/silk_default.yaml
"""

import sys
import os

import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle

from src.models.silk import create_silk_model
from src.data_utils.lafan_dataset import LAFANDataset
from src.evaluation.metrics import compute_metrics, print_metrics_table, compare_with_paper
from src.evaluation.reconstruction import reconstruct_from_model_output


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(checkpoint_path, model):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', None)

    print(f"Loaded checkpoint from epoch {epoch}")
    if best_loss is not None:
        print(f"Best loss: {best_loss:.6f}")

    return model


def create_evaluation_sequence(
    sequence,
    context_frames,
    transition_length,
    stats,
    device
):
    """
    Create evaluation sequence with specific transition length.

    Args:
        sequence: Dict from dataset with 'input' and 'output' keys
        context_frames: Number of context frames (e.g., 10)
        transition_length: Length of transition to generate (e.g., 5, 15, 30, 45)
        stats: Training statistics for normalization
        device: torch device

    Returns:
        input_seq: Input tensor (1, seq_len, din) - on device
        target_seq: Target tensor (1, seq_len, dout) - on device
        mask: Mask tensor (1, seq_len) - on device
    """
    # Get full sequence
    input_features = sequence['input']   # (window_size, din)
    output_features = sequence['output']  # (window_size, dout)

    # Create sequence: [context] + [transition] + [target keyframe]
    seq_len = context_frames + transition_length + 1

    # Ensure we have enough frames
    if seq_len > len(input_features):
        raise ValueError(f"Sequence too short: need {seq_len} frames, have {len(input_features)}")

    # Extract subsequence
    input_seq = input_features[:seq_len].clone()   # (seq_len, din)
    target_seq = output_features[:seq_len].clone()  # (seq_len, dout)

    # Zero-fill transition frames in input
    # Keep: context frames (0 to context_frames-1) and target keyframe (seq_len-1)
    # Zero: transition frames (context_frames to seq_len-2)
    input_seq[context_frames:seq_len-1] = 0.0

    # Create mask: True for transition frames only
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[context_frames:seq_len-1] = True

    # Add batch dimension and move to device
    input_seq = input_seq.unsqueeze(0).to(device)      # (1, seq_len, din)
    target_seq = target_seq.unsqueeze(0).to(device)    # (1, seq_len, dout)
    mask = mask.unsqueeze(0).to(device)                # (1, seq_len)

    return input_seq, target_seq, mask


def evaluate_at_length(
    model,
    test_dataset,
    context_frames,
    transition_length,
    stats,
    device,
    num_samples=None
):
    """
    Evaluate model at a specific transition length.

    Args:
        model: SILK model (in eval mode)
        test_dataset: LAFANDataset instance
        context_frames: Number of context frames
        transition_length: Transition length to evaluate
        stats: Training statistics
        device: torch device
        num_samples: Max number of sequences to evaluate (None = all)

    Returns:
        metrics: Dict with 'l2q', 'l2p', 'npss' values
    """
    model.eval()

    # Collect all predictions and ground truth
    all_pred_quats = []
    all_pred_pos = []
    all_gt_quats = []
    all_gt_pos = []

    # Determine number of sequences to evaluate
    n_sequences = len(test_dataset) if num_samples is None else min(num_samples, len(test_dataset))

    with torch.no_grad():
        for idx in tqdm(range(n_sequences), desc=f'Evaluating length={transition_length}'):
            # Get sequence
            sequence = test_dataset[idx]

            try:
                # Create evaluation input
                input_seq, target_seq, mask = create_evaluation_sequence(
                    sequence,
                    context_frames,
                    transition_length,
                    stats,
                    device
                )
            except ValueError:
                # Skip sequences that are too short
                continue

            # Forward pass
            pred_seq = model(input_seq)  # (1, seq_len, dout)

            # Extract only transition frames for evaluation
            # Start at context_frames, end at seq_len-1 (excluding target keyframe)
            trans_start = context_frames
            trans_end = context_frames + transition_length

            pred_trans = pred_seq[:, trans_start:trans_end, :]  # (1, transition_length, dout)
            target_trans = target_seq[:, trans_start:trans_end, :]  # (1, transition_length, dout)

            # Reconstruct global coordinates
            pred_quats, pred_pos = reconstruct_from_model_output(
                pred_trans,
                stats,
                num_joints=22
            )

            gt_quats, gt_pos = reconstruct_from_model_output(
                target_trans,
                stats,
                num_joints=22
            )

            # Collect
            all_pred_quats.append(pred_quats)
            all_pred_pos.append(pred_pos)
            all_gt_quats.append(gt_quats)
            all_gt_pos.append(gt_pos)

    # Stack all predictions
    pred_quats_batch = np.concatenate(all_pred_quats, axis=0)  # (N, transition_length, 22, 4)
    pred_pos_batch = np.concatenate(all_pred_pos, axis=0)      # (N, transition_length, 22, 3)
    gt_quats_batch = np.concatenate(all_gt_quats, axis=0)      # (N, transition_length, 22, 4)
    gt_pos_batch = np.concatenate(all_gt_pos, axis=0)          # (N, transition_length, 22, 3)

    print(f"  Evaluated {pred_quats_batch.shape[0]} sequences")

    # Compute metrics
    # L2P uses normalized positions (matching LAFAN1 baseline)
    x_mean = stats.get('x_mean', None)
    x_std = stats.get('x_std', None)

    metrics = compute_metrics(
        pred_quats_batch,
        pred_pos_batch,
        gt_quats_batch,
        gt_pos_batch,
        x_mean=x_mean,
        x_std=x_std
    )

    return metrics


def main(args):
    """Main evaluation function."""
    print("=" * 70)
    print("SILK EVALUATION")
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
    if not stats_path.exists():
        print(f"\nError: Training statistics not found at {stats_path}")
        print("Please train the model first or provide correct checkpoint directory.")
        return

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print(f"\nLoaded training statistics from: {stats_path}")

    # Create test dataset (no zero-filling, we'll do it manually)
    print("\n" + "=" * 70)
    print("Loading Test Dataset")
    print("=" * 70)

    test_dataset = LAFANDataset(
        bvh_folder=config['data']['dataset_path'],
        actors=config['data']['test_actors'],
        window_size=65,
        offset=40,
        context_frames=config['data']['context_frames'],
        train_stats=stats,  # Use training stats for normalization
        cache_path='data/cache/test_eval.pkl'
    )

    # Create model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    model = create_silk_model(num_joints=config['data']['num_joints'])
    model = load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    # Evaluate at different transition lengths
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)

    transition_lengths = args.lengths if args.lengths else [5, 15, 30, 45]
    context_frames = config['data']['context_frames']

    results = {}

    for trans_len in transition_lengths:
        print(f"\n{'='*70}")
        print(f"Evaluating at transition length = {trans_len}")
        print(f"{'='*70}")

        metrics = evaluate_at_length(
            model=model,
            test_dataset=test_dataset,
            context_frames=context_frames,
            transition_length=trans_len,
            stats=stats,
            device=device,
            num_samples=args.num_samples
        )

        results[trans_len] = metrics

        print(f"\nResults for length {trans_len}:")
        print(f"  L2Q:  {metrics['l2q']:.4f}")
        print(f"  L2P:  {metrics['l2p']:.4f}")
        print(f"  NPSS: {metrics['npss']:.4f}")

    # Print final results table
    print_metrics_table(results, model_name="SILK")

    # Compare with paper
    compare_with_paper(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SILK model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., checkpoints/best_model.pth)')
    parser.add_argument('--config', type=str, default='configs/silk_default.yaml',
                        help='Path to config file')
    parser.add_argument('--lengths', type=int, nargs='+', default=None,
                        help='Transition lengths to evaluate (default: 5 15 30 45)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Max number of test sequences to evaluate (default: all)')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results.pkl',
                        help='Path to save results')

    args = parser.parse_args()
    main(args)
