#!/usr/bin/env python3
"""
Sampling script for Tiny Diffusion Model.

Usage:
    # Sample with DDPM (default)
    python sample.py --checkpoint checkpoints/model.pt

    # Sample with DDIM (faster)
    python sample.py --checkpoint checkpoints/model.pt --method ddim --ddim_steps 50

    # Generate more samples
    python sample.py --checkpoint checkpoints/model.pt --n_samples 64

    # Save to specific directory
    python sample.py --checkpoint checkpoints/model.pt --output_dir my_samples/

    # Full example
    python sample.py \\
        --checkpoint checkpoints/cifar_model.pt \\
        --method ddim \\
        --ddim_steps 25 \\
        --n_samples 32 \\
        --output_dir samples/cifar/ \\
        --seed 42
"""

import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torchvision.utils import save_image, make_grid

from config import Config, add_common_args
from modules import TinyUNet
from scheduler import NoiseScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate samples from a trained diffusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser = add_common_args(parser)
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Sampling options
    parser.add_argument("--method", type=str, default="ddpm",
                        choices=["ddpm", "ddim"],
                        help="Sampling method")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta parameter (0=deterministic, 1=stochastic)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="samples",
                        help="Directory to save samples")
    parser.add_argument("--save_grid", action="store_true", default=True,
                        help="Save samples as image grid")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save each sample as individual file")
    parser.add_argument("--show", action="store_true",
                        help="Display samples (requires display)")
    
    return parser.parse_args()


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    scheduler: NoiseScheduler,
    n_samples: int,
    img_size: int,
    channels: int,
    device: torch.device,
    show_progress: bool = True
) -> torch.Tensor:
    """Standard DDPM sampling (T steps)."""
    model.eval()
    
    x = torch.randn((n_samples, channels, img_size, img_size), device=device)
    
    timesteps = range(scheduler.n_timesteps - 1, -1, -1)
    if show_progress:
        timesteps = tqdm(timesteps, desc="DDPM Sampling", leave=False)
    
    for t in timesteps:
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        predicted_noise = model(x, t_batch)
        
        alpha = scheduler.alphas[t]
        alpha_hat = scheduler.alpha_hats[t]
        beta = scheduler.betas[t]
        
        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_hat)) * predicted_noise
        )
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(scheduler.posterior_variance[t])
            x = mean + sigma * noise
        else:
            x = mean
    
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def sample_ddim(
    model: torch.nn.Module,
    scheduler: NoiseScheduler,
    n_samples: int,
    img_size: int,
    channels: int,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    show_progress: bool = True
) -> torch.Tensor:
    """DDIM sampling (accelerated)."""
    model.eval()
    
    x = torch.randn((n_samples, channels, img_size, img_size), device=device)
    
    step_ratio = scheduler.n_timesteps // steps
    timesteps = list(range(0, scheduler.n_timesteps, step_ratio))[::-1]
    
    iterator = timesteps
    if show_progress:
        iterator = tqdm(timesteps, desc=f"DDIM Sampling ({steps} steps)", leave=False)
    
    for i, t in enumerate(iterator):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        predicted_noise = model(x, t_batch)
        
        alpha_hat_t = scheduler.alpha_hats[t]
        
        pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_hat_t_prev = scheduler.alpha_hats[t_prev]
        else:
            alpha_hat_t_prev = torch.tensor(1.0, device=device)
        
        sigma_t = eta * torch.sqrt(
            (1 - alpha_hat_t_prev) / (1 - alpha_hat_t) *
            (1 - alpha_hat_t / alpha_hat_t_prev)
        ) if eta > 0 else 0
        
        dir_xt = torch.sqrt(1 - alpha_hat_t_prev - sigma_t**2) * predicted_noise
        
        noise = torch.randn_like(x) if (i < len(timesteps) - 1 and eta > 0) else 0
        
        x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt + sigma_t * noise
    
    return (x.clamp(-1, 1) + 1) / 2


def save_samples(
    samples: torch.Tensor,
    output_dir: Path,
    prefix: str = "sample",
    save_grid: bool = True,
    save_individual: bool = False,
    show: bool = False
):
    """Save generated samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = len(samples)
    nrow = int(n_samples ** 0.5)
    
    if save_grid:
        grid_path = output_dir / f"{prefix}_grid.png"
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        save_image(grid, grid_path)
        print(f"✅ Grid saved to {grid_path}")
    
    if save_individual:
        for i, img in enumerate(samples):
            img_path = output_dir / f"{prefix}_{i:04d}.png"
            save_image(img, img_path)
        print(f"✅ {n_samples} individual images saved to {output_dir}")
    
    if show:
        fig, axes = plt.subplots(nrow, (n_samples + nrow - 1) // nrow, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < n_samples:
                img = samples[i].cpu().permute(1, 2, 0).numpy()
                if img.shape[2] == 1:
                    img = img.squeeze(-1)
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                else:
                    ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    args = parse_args()
    
    device = torch.device(args.device)
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract config from checkpoint or use CLI args
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        img_size = ckpt_config.get('img_size', args.img_size)
        channels = ckpt_config.get('channels', args.channels)
        base_channels = ckpt_config.get('base_channels', args.base_channels)
        n_timesteps = ckpt_config.get('n_timesteps', args.n_timesteps)
        schedule_type = ckpt_config.get('schedule_type', args.schedule_type)
        model_state = checkpoint['model_state_dict']
    else:
        img_size = args.img_size
        channels = args.channels
        base_channels = args.base_channels
        n_timesteps = args.n_timesteps
        schedule_type = args.schedule_type
        model_state = checkpoint
    
    print(f"Configuration:")
    print(f"  - Image size: {img_size}x{img_size}")
    print(f"  - Channels: {channels}")
    print(f"  - Timesteps: {n_timesteps}")
    print(f"  - Method: {args.method}")
    if args.method == "ddim":
        print(f"  - DDIM steps: {args.ddim_steps}")
        print(f"  - DDIM eta: {args.ddim_eta}")
    
    # Create model and load weights
    model = TinyUNet(
        img_size=img_size,
        channels=channels,
        base_channels=base_channels
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Create scheduler
    scheduler = NoiseScheduler(
        n_timesteps=n_timesteps,
        schedule_type=schedule_type,
        device=str(device)
    )
    
    # Generate samples
    print(f"\nGenerating {args.n_samples} samples...")
    
    all_samples = []
    remaining = args.n_samples
    
    with torch.no_grad():
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            
            if args.method == "ddpm":
                samples = sample_ddpm(
                    model, scheduler, batch_size, img_size, channels, device
                )
            else:
                samples = sample_ddim(
                    model, scheduler, batch_size, img_size, channels, device,
                    steps=args.ddim_steps, eta=args.ddim_eta
                )
            
            all_samples.append(samples)
            remaining -= batch_size
    
    all_samples = torch.cat(all_samples, dim=0)
    
    # Save samples
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.method}_{timestamp}"
    
    save_samples(
        all_samples,
        output_dir,
        prefix=prefix,
        save_grid=args.save_grid,
        save_individual=args.save_individual,
        show=args.show
    )
    
    print(f"\n✅ Done! Generated {args.n_samples} samples.")


if __name__ == "__main__":
    main()