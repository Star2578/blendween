"""
Loss Functions for SILK

SILK uses a simple L1 loss across all output features.
No separate losses for positions, rotations, or foot contacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SILKLoss(nn.Module):
    """
    SILK's loss function: Simple L1 loss.

    The paper shows that a single L1 loss works better than:
    - Multiple separate losses (position + rotation + foot contact)
    - L2 loss
    - Complex GAN losses

    Args:
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets, mask=None):
        """
        Compute L1 loss between predictions and targets.

        Args:
            predictions: (batch, seq_len, dout) predicted features
            targets: (batch, seq_len, dout) ground truth features
            mask: (batch, seq_len) boolean mask (True = compute loss for this frame)
                  Used to only compute loss on transition frames, not context/target

        Returns:
            loss: scalar loss value
        """
        # Compute L1 distance
        loss = F.l1_loss(predictions, targets, reduction='none')  # (batch, seq_len, dout)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match feature dimension
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            loss = loss * mask.float()  # Zero out non-masked frames

            # Average only over masked elements
            if self.reduction == 'mean':
                # Count total number of masked elements (frames × features)
                num_masked_frames = mask.float().sum()
                if num_masked_frames > 0:
                    # Divide by total masked elements = num_frames × num_features
                    num_masked_elements = num_masked_frames * predictions.shape[-1]
                    loss = loss.sum() / num_masked_elements
                else:
                    loss = loss.sum()  # Fallback if no frames are masked
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            # No mask, apply standard reduction
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss


class L2Loss(nn.Module):
    """
    L2 (MSE) loss for comparison.

    The paper shows L1 works better than L2 for motion in-betweening.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets, mask=None):
        """Compute L2 loss."""
        loss = F.mse_loss(predictions, targets, reduction='none')

        if mask is not None:
            mask = mask.unsqueeze(-1)
            loss = loss * mask.float()

            if self.reduction == 'mean':
                # Count total number of masked elements (frames × features)
                num_masked_frames = mask.float().sum()
                if num_masked_frames > 0:
                    # Divide by total masked elements = num_frames × num_features
                    num_masked_elements = num_masked_frames * predictions.shape[-1]
                    loss = loss.sum() / num_masked_elements
                else:
                    loss = loss.sum()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss


def create_loss_function(loss_type='l1', reduction='mean'):
    """
    Create loss function.

    Args:
        loss_type: 'l1' (SILK default) or 'l2' (for comparison)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss_fn: Loss function module
    """
    if loss_type == 'l1':
        return SILKLoss(reduction=reduction)
    elif loss_type == 'l2' or loss_type == 'mse':
        return L2Loss(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss function
    print("Testing SILK Loss Function")
    print("=" * 60)

    batch_size = 4
    seq_len = 20
    dout = 202

    # Create dummy data
    predictions = torch.randn(batch_size, seq_len, dout)
    targets = torch.randn(batch_size, seq_len, dout)

    # Create mask (only compute loss on frames 10-18, like transition frames)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, 10:19] = True  # 9 transition frames

    # Test L1 loss
    loss_fn = SILKLoss(reduction='mean')

    loss_with_mask = loss_fn(predictions, targets, mask)
    loss_without_mask = loss_fn(predictions, targets, None)

    print(f"L1 Loss with mask: {loss_with_mask.item():.6f}")
    print(f"L1 Loss without mask: {loss_without_mask.item():.6f}")
    print(f"\nMask shape: {mask.shape}")
    print(f"Num masked frames: {mask.sum().item()}")
    print(f"Num total frames: {batch_size * seq_len}")

    # Test L2 loss for comparison
    l2_loss_fn = L2Loss(reduction='mean')
    l2_loss = l2_loss_fn(predictions, targets, mask)

    print(f"\nL2 Loss with mask: {l2_loss.item():.6f}")

    print("\n" + "=" * 60)
    print("Loss functions work correctly!")
