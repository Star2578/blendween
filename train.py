"""
SILK Training Script

Train the SILK model for motion in-betweening on LAFAN1 dataset.

Usage:
    python train.py --config configs/silk_default.yaml
"""

import sys
import os

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from pathlib import Path

from src.models.silk import create_silk_model
from src.data.lafan_dataset import create_dataloaders
from src.training.losses import create_loss_function
from src.training.scheduler import create_scheduler


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, checkpoint_dir, is_best=False, is_regular=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
    }

    if is_regular:
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"New best model saved: {best_path}")

    # Save latest (for easy resuming)
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Best loss: {best_loss:.6f}")

    return epoch, best_loss


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, writer, log_every=10, grad_clip=1.0):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        inputs = batch['input'].to(device)          # (batch, seq_len, din)
        targets = batch['target'].to(device)        # (batch, seq_len, dout)
        mask = batch['mask'].to(device)             # (batch, seq_len)

        # Forward pass
        predictions = model(inputs)  # (batch, seq_len, dout)

        # Compute loss (only on transition frames)
        loss = criterion(predictions, targets, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })

        # Log to TensorBoard
        if batch_idx % log_every == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', current_lr, global_step)

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', leave=False):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            mask = batch['mask'].to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets, mask)

            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    print("=" * 70)
    print("SILK TRAINING")
    print("=" * 70)
    print(f"Configuration: {args.config}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.get('seed', 42))

    # Create directories
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Create dataloaders
    print("\n" + "=" * 70)
    print("Loading Dataset")
    print("=" * 70)

    train_loader, val_loader, train_stats = create_dataloaders(
        bvh_folder=config['data']['dataset_path'],
        train_actors=config['data']['train_actors'],
        test_actors=config['data']['test_actors'],
        batch_size=config['training']['batch_size'],
        window_size=config['data']['window_size'],
        offset=config['data']['offset'],
        context_frames=config['data']['context_frames'],
        min_transition=config['data']['min_transition'],
        max_transition=config['data']['max_transition'],
        num_workers=config['training']['num_workers']
    )

    # Save training statistics
    stats_path = checkpoint_dir / 'train_stats.pkl'
    import pickle
    with open(stats_path, 'wb') as f:
        pickle.dump(train_stats, f)
    print(f"\nTraining statistics saved to: {stats_path}")

    # Create model
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)

    model = create_silk_model(
        num_joints=config['data']['num_joints'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)

    # Create loss function
    criterion = create_loss_function(
        loss_type=config['training']['loss_type'],
        reduction='mean'
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        eps=config['training']['eps'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='noam',
        d_model=config['model']['d_model'],
        warmup_steps=config['training']['warmup_steps']
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    print(f"Epochs: {start_epoch} -> {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}")

    for epoch in range(start_epoch, config['training']['num_epochs']):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, writer, log_every=config['logging']['log_every'],
            grad_clip=config['training']['grad_clip']
        )

        # Validate
        if (epoch + 1) % config['logging']['eval_every'] == 0:
            val_loss = validate(model, val_loader, criterion, device)

            # Log to TensorBoard
            writer.add_scalar('train/loss_epoch', train_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)

            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            is_regular = (epoch + 1) % config['logging']['save_every'] == 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                best_val_loss, checkpoint_dir, is_best=is_best, is_regular=is_regular
            )

        else:
            # Just log training loss
            writer.add_scalar('train/loss_epoch', train_loss, epoch)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s")

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, config['training']['num_epochs'] - 1,
        best_val_loss, checkpoint_dir
    )

    writer.close()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SILK model')
    parser.add_argument('--config', type=str, default='configs/silk_default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    main(args)
