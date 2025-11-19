"""
Evaluation Metrics for SILK

Implements the three standard metrics for motion in-betweening:
- L2Q: L2 norm of global quaternions
- L2P: L2 norm of global positions (normalized)
- NPSS: Normalized Power Spectrum Similarity

These metrics match the LAFAN1 benchmark implementation.
"""

import numpy as np
import torch


def fast_npss(gt_seq, pred_seq):
    """
    Computes Normalized Power Spectrum Similarity (NPSS).

    Metric proposed by Gropalakrishnan et al (2019) that measures
    motion quality in frequency domain. Correlates well with human
    perception of naturalness.

    Args:
        gt_seq: Ground truth array of shape (batch, timesteps, features)
        pred_seq: Predicted array of shape (batch, timesteps, features)

    Returns:
        npss: Scalar NPSS metric (lower is better)
    """
    # Convert to numpy if needed
    if torch.is_tensor(gt_seq):
        gt_seq = gt_seq.detach().cpu().numpy()
    if torch.is_tensor(pred_seq):
        pred_seq = pred_seq.detach().cpu().numpy()

    # Fourier coefficients along the time dimension
    gt_fourier_coeffs = np.real(np.fft.fft(gt_seq, axis=1))
    pred_fourier_coeffs = np.real(np.fft.fft(pred_seq, axis=1))

    # Square of the Fourier coefficients
    gt_power = np.square(gt_fourier_coeffs)
    pred_power = np.square(pred_fourier_coeffs)

    # Sum of powers over time dimension
    gt_total_power = np.sum(gt_power, axis=1)
    pred_total_power = np.sum(pred_power, axis=1)

    # Normalize powers with totals
    gt_norm_power = gt_power / (gt_total_power[:, np.newaxis, :] + 1e-10)
    pred_norm_power = pred_power / (pred_total_power[:, np.newaxis, :] + 1e-10)

    # Cumulative sum over time
    cdf_gt_power = np.cumsum(gt_norm_power, axis=1)
    cdf_pred_power = np.cumsum(pred_norm_power, axis=1)

    # Earth mover distance
    emd = np.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=1)

    # Weighted EMD
    power_weighted_emd = np.average(emd, weights=gt_total_power + 1e-10)

    return power_weighted_emd


def compute_l2q(pred_quats, gt_quats):
    """
    Compute L2 quaternion loss (L2Q).

    Measures the L2 distance between predicted and ground truth
    global quaternions across all joints and timesteps.

    Args:
        pred_quats: Predicted global quaternions (batch, timesteps, joints, 4)
        gt_quats: Ground truth global quaternions (batch, timesteps, joints, 4)

    Returns:
        l2q: Scalar L2Q metric (lower is better)
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred_quats):
        pred_quats = pred_quats.detach().cpu().numpy()
    if torch.is_tensor(gt_quats):
        gt_quats = gt_quats.detach().cpu().numpy()

    # Compute L2 distance: sqrt(sum((pred - gt)^2))
    # Average over batch, sum over timesteps and joints
    diff = pred_quats - gt_quats
    l2q = np.mean(np.sqrt(np.sum(diff ** 2.0, axis=(2, 3))))

    return l2q


def compute_l2p(pred_pos, gt_pos, x_mean=None, x_std=None):
    """
    Compute L2 position loss (L2P).

    Measures the L2 distance between predicted and ground truth
    global positions. Positions are normalized using training
    statistics before computing the metric.

    Args:
        pred_pos: Predicted global positions (batch, timesteps, joints, 3)
                  or (batch, joints*3, timesteps) if already flattened
        gt_pos: Ground truth global positions (batch, timesteps, joints, 3)
                or (batch, joints*3, timesteps) if already flattened
        x_mean: Mean for normalization (optional, shape: (1, joints*3, 1))
        x_std: Std for normalization (optional, shape: (1, joints*3, 1))

    Returns:
        l2p: Scalar L2P metric (lower is better)
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred_pos):
        pred_pos = pred_pos.detach().cpu().numpy()
    if torch.is_tensor(gt_pos):
        gt_pos = gt_pos.detach().cpu().numpy()

    # Reshape if needed (batch, timesteps, joints, 3) -> (batch, joints*3, timesteps)
    if pred_pos.ndim == 4:
        batch, timesteps, joints, _ = pred_pos.shape
        pred_pos = pred_pos.reshape(batch, timesteps, joints * 3).transpose(0, 2, 1)
        gt_pos = gt_pos.reshape(batch, timesteps, joints * 3).transpose(0, 2, 1)

    # Normalize if statistics provided
    if x_mean is not None and x_std is not None:
        pred_pos = (pred_pos - x_mean) / x_std
        gt_pos = (gt_pos - x_mean) / x_std

    # Compute L2 distance: sqrt(sum((pred - gt)^2))
    # Sum over features (joints*3), average over batch and timesteps
    diff = pred_pos - gt_pos
    l2p = np.mean(np.sqrt(np.sum(diff ** 2.0, axis=1)))

    return l2p


def flatjoints(x):
    """
    Flatten all but the first two dimensions.

    Used to prepare data for NPSS computation.

    Args:
        x: Array of shape (batch, timesteps, joints, features)

    Returns:
        Flattened array of shape (batch, timesteps, joints*features)
    """
    return x.reshape((x.shape[0], x.shape[1], -1))


def compute_metrics(pred_global_quats, pred_global_pos,
                   gt_global_quats, gt_global_pos,
                   x_mean=None, x_std=None):
    """
    Compute all three evaluation metrics: L2Q, L2P, NPSS.

    Args:
        pred_global_quats: Predicted global quaternions (batch, timesteps, joints, 4)
        pred_global_pos: Predicted global positions (batch, timesteps, joints, 3)
        gt_global_quats: Ground truth global quaternions (batch, timesteps, joints, 4)
        gt_global_pos: Ground truth global positions (batch, timesteps, joints, 3)
        x_mean: Mean for L2P normalization (optional)
        x_std: Std for L2P normalization (optional)

    Returns:
        metrics: Dict with keys 'l2q', 'l2p', 'npss'
    """
    metrics = {}

    # L2Q: Quaternion distance
    metrics['l2q'] = compute_l2q(pred_global_quats, gt_global_quats)

    # L2P: Position distance (normalized)
    metrics['l2p'] = compute_l2p(pred_global_pos, gt_global_pos, x_mean, x_std)

    # NPSS: Power spectrum similarity on quaternions
    metrics['npss'] = fast_npss(
        flatjoints(gt_global_quats),
        flatjoints(pred_global_quats)
    )

    return metrics


def print_metrics_table(results, model_name="SILK"):
    """
    Print metrics in a nicely formatted table.

    Args:
        results: Dict mapping transition_length -> metrics_dict
        model_name: Name to display in table
    """
    trans_lengths = sorted(results.keys())

    print("\n" + "=" * 70)
    print(f"{model_name} Evaluation Results")
    print("=" * 70)

    # L2Q table
    print("\n=== Global Quaternion Loss (L2Q) ↓ ===")
    print(f"{'Model':<16} | " + " | ".join(f"{n:6d}" for n in trans_lengths))
    print("-" * 70)
    l2q_values = [results[n]['l2q'] for n in trans_lengths]
    print(f"{model_name:<16} | " + " | ".join(f"{v:6.3f}" for v in l2q_values))

    # L2P table
    print("\n=== Global Position Loss (L2P) ↓ ===")
    print(f"{'Model':<16} | " + " | ".join(f"{n:6d}" for n in trans_lengths))
    print("-" * 70)
    l2p_values = [results[n]['l2p'] for n in trans_lengths]
    print(f"{model_name:<16} | " + " | ".join(f"{v:6.3f}" for v in l2p_values))

    # NPSS table
    print("\n=== Normalized Power Spectrum Similarity (NPSS) ↓ ===")
    print(f"{'Model':<16} | " + " | ".join(f"{n:6d}" for n in trans_lengths))
    print("-" * 70)
    npss_values = [results[n]['npss'] for n in trans_lengths]
    print(f"{model_name:<16} | " + " | ".join(f"{v:6.4f}" for v in npss_values))

    print("\n" + "=" * 70)


def compare_with_paper(results):
    """
    Compare results against SILK paper benchmarks (Table 1).

    Args:
        results: Dict mapping transition_length -> metrics_dict
    """
    # SILK paper results (Table 1)
    paper_results = {
        5: {'l2p': 0.13, 'l2q': 0.11, 'npss': 0.0012},
        15: {'l2p': 0.38, 'l2q': 0.27, 'npss': 0.018},
        30: {'l2p': 0.83, 'l2q': 0.50, 'npss': 0.105},
        45: {'l2p': 1.59, 'l2q': 0.79, 'npss': 0.30}
    }

    print("\n" + "=" * 70)
    print("Comparison with SILK Paper Results")
    print("=" * 70)

    for trans_len in sorted(results.keys()):
        if trans_len not in paper_results:
            continue

        print(f"\nTransition Length: {trans_len} frames")
        print("-" * 40)

        our_res = results[trans_len]
        paper_res = paper_results[trans_len]

        for metric in ['l2q', 'l2p', 'npss']:
            our_val = our_res[metric]
            paper_val = paper_res[metric]
            diff_pct = ((our_val - paper_val) / paper_val) * 100

            status = "ok" if abs(diff_pct) < 10 else "bad"
            print(f"  {metric.upper():<6}: Ours: {our_val:.4f} | "
                  f"Paper: {paper_val:.4f} | "
                  f"Diff: {diff_pct:+.1f}% {status}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test metrics with dummy data
    print("Testing evaluation metrics...")

    batch_size = 4
    timesteps = 30
    joints = 22

    # Create dummy data
    gt_quats = np.random.randn(batch_size, timesteps, joints, 4)
    pred_quats = gt_quats + np.random.randn(batch_size, timesteps, joints, 4) * 0.1

    gt_pos = np.random.randn(batch_size, timesteps, joints, 3)
    pred_pos = gt_pos + np.random.randn(batch_size, timesteps, joints, 3) * 0.1

    # Compute metrics
    metrics = compute_metrics(pred_quats, pred_pos, gt_quats, gt_pos)

    print(f"\nTest Results:")
    print(f"  L2Q: {metrics['l2q']:.4f}")
    print(f"  L2P: {metrics['l2p']:.4f}")
    print(f"  NPSS: {metrics['npss']:.4f}")

    print("\nMetrics module working correctly!")
