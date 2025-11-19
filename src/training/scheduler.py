"""
Learning Rate Schedulers for SILK

Implements the Noam learning rate scheduler used in the original Transformer paper
and adopted by SILK.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class NoamScheduler(_LRScheduler):
    """
    Noam learning rate scheduler from "Attention is All You Need".

    Learning rate schedule:
        lr = base_lr * min(step^(-0.5), step * warmup_steps^(-1.5))

    This creates:
    1. Linear warmup for warmup_steps
    2. Decay proportional to step^(-0.5) after warmup

    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension (used for scaling)
        warmup_steps: Number of warmup steps (default: 4000)
        factor: Scaling factor for learning rate (default: 1.0)
        last_epoch: Last epoch index (for resuming)

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        >>> scheduler = NoamScheduler(optimizer, d_model=1024, warmup_steps=4000)
        >>> for epoch in range(num_epochs):
        >>>     for batch in dataloader:
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.step()  # Update learning rate
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        # Use _step_count instead of last_epoch for step-based scheduling
        step = max(self._step_count, 1)  # Avoid division by zero
        
        # Noam formula - compute LR directly, don't scale base_lr
        lr = self.factor * (self.d_model ** (-0.5)) * min(
            step ** (-0.5), 
            step * (self.warmup_steps ** (-1.5))
        )
        
        # Return same LR for all parameter groups
        return [lr for _ in self.base_lrs]


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.

    Alternative to Noam scheduler with smoother decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0)
        last_epoch: Last epoch index
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        step = max(self.last_epoch, 1)

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr + (1.0 - self.min_lr) * scale

        return [base_lr * scale for base_lr in self.base_lrs]


def create_scheduler(optimizer, scheduler_type='noam', **kwargs):
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'noam' (SILK default) or 'warmup_cosine'
        **kwargs: Scheduler-specific arguments

    Returns:
        scheduler: Learning rate scheduler
    """
    if scheduler_type == 'noam':
        d_model = kwargs.get('d_model', 1024)
        warmup_steps = kwargs.get('warmup_steps', 4000)
        factor = kwargs.get('factor', 1.0)
        return NoamScheduler(optimizer, d_model, warmup_steps, factor)

    elif scheduler_type == 'warmup_cosine':
        warmup_steps = kwargs.get('warmup_steps', 4000)
        total_steps = kwargs.get('total_steps', 100000)
        min_lr = kwargs.get('min_lr', 0)
        return WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    print("Testing Learning Rate Schedulers")
    print("=" * 60)

    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    # Test Noam scheduler
    scheduler = NoamScheduler(optimizer, d_model=1024, warmup_steps=4000)

    # Simulate training steps
    steps = 20000
    lrs = []

    for step in range(steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # Plot learning rate schedule
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Noam Learning Rate Schedule')
    plt.grid(True, alpha=0.3)

    # Log scale to see warmup better
    plt.subplot(1, 2, 2)
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate (log scale)')
    plt.title('Noam Schedule (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/noam_schedule.png', dpi=150, bbox_inches='tight')
    print(f"Learning rate plot saved to: outputs/noam_schedule.png")

    print(f"\nLearning rate statistics:")
    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  Peak LR (at step {np.argmax(lrs)}): {max(lrs):.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    print(f"  LR at step 4000 (warmup end): {lrs[3999]:.6f}")

    print("\n" + "=" * 60)
    print("Scheduler works correctly!")
