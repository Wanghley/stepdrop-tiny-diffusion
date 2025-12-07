#!/usr/bin/env python3
"""
Training script for Tiny Diffusion Model.

Usage:
    # Train on MNIST (default)
    python train.py

    # Train on CIFAR-10
    python train.py --dataset cifar10 --img_size 32 --channels 3 --epochs 50

    # Train on custom dataset
    python train.py --dataset custom --custom_data_dir /path/to/images --img_size 64 --channels 3

    # Resume training
    python train.py --resume checkpoints/model.pt

    # Full example with all options
    python train.py \\
        --dataset cifar10 \\
        --batch_size 64 \\
        --epochs 100 \\
        --lr 1e-4 \\
        --base_channels 128 \\
        --save_path checkpoints/cifar_model.pt \\
        --seed 42
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from config import Config, add_common_args
from modules import get_model
from scheduler import NoiseScheduler
from dataset import get_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Tiny Diffusion Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser = add_common_args(parser)
    
    # Training-specific args
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for logs")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log every N batches")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (overrides CLI args)")
    
    return parser.parse_args()


def train(config: Config):
    """Main training function."""
    device = config.get_device()
    
    # Set seed
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {config.dataset}")
    print(f"Image size: {config.img_size}x{config.img_size}")
    print(f"Channels: {config.channels}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Timesteps: {config.n_timesteps}")
    print(f"Schedule: {config.schedule_type}")
    print("=" * 60)
    
    # Create directories
    Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.to_json(Path(config.log_dir) / "config.json")
    
    # Initialize components
    dataloader = get_dataloader(
        dataset=config.dataset,
        batch_size=config.batch_size,
        img_size=config.img_size,
        channels=config.channels,
        data_dir=config.data_dir,
        custom_data_dir=config.custom_data_dir,
        num_workers=config.num_workers
    )
    
    model = get_model(
        img_size=config.img_size,
        channels=config.channels,
        base_channels=config.base_channels,
        time_emb_dim=config.time_emb_dim,
        device=str(device)
    )
    
    scheduler = NoiseScheduler(
        n_timesteps=config.n_timesteps,
        schedule_type=config.schedule_type,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=str(device)
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs * len(dataloader), 
        eta_min=1e-6
    )
    
    loss_fn = nn.MSELoss()
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    if config.resume:
        print(f"Resuming from {config.resume}")
        checkpoint = torch.load(config.resume, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
        else:
            model.load_state_dict(checkpoint)
    
    # Training loop
    history = {'train_loss': [], 'lr': []}
    
    model.train()
    for epoch in range(start_epoch, config.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Sample timesteps
            t = scheduler.sample_timesteps(images.shape[0])
            
            # Add noise
            x_t, noise = scheduler.noise_images(images, t)
            
            # Predict noise
            predicted_noise = model(x_t, t)
            
            # Compute loss
            loss = loss_fn(predicted_noise, noise)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = epoch_loss / num_batches
        current_lr = lr_scheduler.get_last_lr()[0]
        
        history['train_loss'].append(avg_loss)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': config.__dict__
            }, config.save_path)
            print(f"✅ Best model saved to {config.save_path}")
        
        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            ckpt_path = Path(config.save_path).parent / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': config.__dict__
            }, ckpt_path)
            print(f"📁 Checkpoint saved to {ckpt_path}")
    
    # Save training history
    with open(Path(config.log_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {config.save_path}")
    print("=" * 60)
    
    return model, history


def main():
    args = parse_args()
    
    # Load from config file if specified
    if args.config:
        config = Config.from_json(args.config)
        # Override with CLI args
        for key, value in vars(args).items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
    else:
        config = Config.from_args(args)
    
    # Handle resume
    if args.resume:
        config.resume = args.resume
    
    train(config)


if __name__ == "__main__":
    main()