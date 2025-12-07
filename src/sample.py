#!/usr/bin/env python3
"""
Sampling script for Tiny Diffusion Model.

Usage:
    # Sample with DDPM (default)
    python sample.py --checkpoint checkpoints/model.pt

    # Sample with DDIM (faster)
    python sample.py --checkpoint checkpoints/model.pt --method ddim --ddim_steps 50

    # Sample with StepDrop
    python sample.py --checkpoint checkpoints/model.pt --method stepdrop --skip_prob 0.3

    # Generate more samples
    python sample.py --checkpoint checkpoints/model.pt --n_samples 64

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
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torchvision.utils import save_image, make_grid

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, add_common_args
from src.modules import TinyUNet
from src.sampler import DDPMSampler, DDIMSampler, StepDropSampler, AdaptiveStepDropSampler


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
                        choices=["ddpm", "ddim", "stepdrop", "adaptive_stepdrop"],
                        help="Sampling method")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta parameter (0=deterministic, 1=stochastic)")
    parser.add_argument("--skip_prob", type=float, default=0.3,
                        help="StepDrop skip probability")
    parser.add_argument("--skip_strategy", type=str, default="linear",
                        choices=["constant", "linear", "cosine_sq", "quadratic", 
                                 "early_skip", "late_skip", "critical_preserve"],
                        help="StepDrop skip strategy")
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
    
    # Normalize samples from [-1, 1] to [0, 1] if needed
    if samples.min() < 0:
        samples = (samples.clamp(-1, 1) + 1) / 2
    
    if save_grid:
        grid_path = output_dir / f"{prefix}_grid.png"
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        save_image(grid, grid_path)
        print(f"[SUCCESS] Grid saved to {grid_path}")
    
    if save_individual:
        for i, img in enumerate(samples):
            img_path = output_dir / f"{prefix}_{i:04d}.png"
            save_image(img, img_path)
        print(f"[SUCCESS] {n_samples} individual images saved to {output_dir}")
    
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
        model_state = checkpoint['model_state_dict']
    else:
        img_size = args.img_size
        channels = args.channels
        base_channels = args.base_channels
        n_timesteps = args.n_timesteps
        model_state = checkpoint
    
    print(f"Configuration:")
    print(f"  - Image size: {img_size}x{img_size}")
    print(f"  - Channels: {channels}")
    print(f"  - Timesteps: {n_timesteps}")
    print(f"  - Method: {args.method}")
    
    if args.method == "ddim":
        print(f"  - DDIM steps: {args.ddim_steps}")
        print(f"  - DDIM eta: {args.ddim_eta}")
    elif args.method in ["stepdrop", "adaptive_stepdrop"]:
        print(f"  - Skip probability: {args.skip_prob}")
        print(f"  - Skip strategy: {args.skip_strategy}")
    
    # Create model and load weights
    model = TinyUNet(
        img_size=img_size,
        channels=channels,
        base_channels=base_channels
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Create sampler based on method
    if args.method == "ddpm":
        print(f"Initializing DDPM sampler with {n_timesteps} timesteps")
        sampler = DDPMSampler(num_timesteps=n_timesteps)
    elif args.method == "ddim":
        print(f"Initializing DDIM sampler with {n_timesteps} timesteps, {args.ddim_steps} inference steps")
        sampler = DDIMSampler(
            num_timesteps=n_timesteps,
            num_inference_steps=args.ddim_steps,
            eta=args.ddim_eta
        )
    elif args.method == "stepdrop":
        print(f"Initializing StepDrop sampler with {n_timesteps} timesteps")
        sampler = StepDropSampler(num_timesteps=n_timesteps)
    elif args.method == "adaptive_stepdrop":
        print(f"Initializing Adaptive StepDrop sampler with {n_timesteps} timesteps")
        sampler = AdaptiveStepDropSampler(num_timesteps=n_timesteps)
    
    # Generate samples
    print(f"\nGenerating {args.n_samples} samples...")
    
    all_samples = []
    remaining = args.n_samples
    total_stats = {"steps_taken": 0, "steps_skipped": 0}
    
    with torch.no_grad():
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            shape = (batch_size, channels, img_size, img_size)
            
            if args.method == "ddpm":
                samples = sampler.sample(
                    model=model,
                    shape=shape,
                    device=str(device),
                    show_progress=(remaining == args.n_samples)
                )
            elif args.method == "ddim":
                samples = sampler.sample(
                    model=model,
                    shape=shape,
                    device=str(device),
                    show_progress=(remaining == args.n_samples)
                )
            elif args.method == "stepdrop":
                samples, stats = sampler.sample(
                    model=model,
                    shape=shape,
                    device=str(device),
                    skip_prob=args.skip_prob,
                    skip_strategy=args.skip_strategy,
                    show_progress=(remaining == args.n_samples)
                )
                if stats:
                    total_stats["steps_taken"] += stats.steps_taken
                    total_stats["steps_skipped"] += stats.steps_skipped
            elif args.method == "adaptive_stepdrop":
                samples, stats = sampler.sample(
                    model=model,
                    shape=shape,
                    device=str(device),
                    base_skip_prob=args.skip_prob,
                    show_progress=(remaining == args.n_samples)
                )
                if stats:
                    total_stats["steps_taken"] += stats["steps_taken"]
                    total_stats["steps_skipped"] += stats["steps_skipped"]
            
            # Normalize from [-1, 1] to [0, 1]
            samples = (samples.clamp(-1, 1) + 1) / 2
            
            all_samples.append(samples)
            remaining -= batch_size
            print(f"  Generated {args.n_samples - remaining}/{args.n_samples}")
    
    all_samples = torch.cat(all_samples, dim=0)
    
    # Print stats for StepDrop methods
    if args.method in ["stepdrop", "adaptive_stepdrop"]:
        batches = (args.n_samples + args.batch_size - 1) // args.batch_size
        avg_steps = total_stats["steps_taken"] / batches
        avg_skipped = total_stats["steps_skipped"] / batches
        print(f"\nStepDrop Stats:")
        print(f"  - Average steps taken: {avg_steps:.0f}")
        print(f"  - Average steps skipped: {avg_skipped:.0f}")
        print(f"  - Effective skip rate: {avg_skipped / n_timesteps:.1%}")
    
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
    
    print(f"\n[SUCCESS] Done! Generated {args.n_samples} samples.")


if __name__ == "__main__":
    main()