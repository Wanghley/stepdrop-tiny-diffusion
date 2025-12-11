#!/usr/bin/env python3
"""
Simple StepDrop Demo (No Dependencies)
======================================

A minimal demo showing noise → image generation.
Supports CUDA, MPS (Apple Silicon), and CPU.

Usage:
    python scripts/demo_simple.py
    python scripts/demo_simple.py --checkpoint checkpoints/model.pt
    python scripts/demo_simple.py --seed 123 --method ddim
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules import TinyUNet
from src.sampler import DDPMSampler, DDIMSampler, StepDropSampler, TargetNFEStepDropSampler


def get_best_device() -> str:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_info(device: str) -> str:
    """Get human-readable device info."""
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        return f"CUDA ({name})"
    elif device == "mps":
        return "MPS (Apple Silicon)"
    else:
        return "CPU"


def load_model(checkpoint_path, device):
    """Load or create model."""
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📦 Loading: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(ckpt, dict) and 'config' in ckpt:
            cfg = ckpt['config']
            model = TinyUNet(
                img_size=cfg.get('img_size', 32),
                channels=cfg.get('channels', 3),
                base_channels=cfg.get('base_channels', 64)
            )
            model.load_state_dict(ckpt['model_state_dict'])
            channels = cfg.get('channels', 3)
            img_size = cfg.get('img_size', 32)
        else:
            model = TinyUNet(img_size=32, channels=3, base_channels=64)
            state = ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state)
            channels, img_size = 3, 32
        
        model.to(device).eval()
    else:
        print("⚠️  No checkpoint - using untrained model (outputs will be noisy)")
        model = TinyUNet(img_size=32, channels=3, base_channels=32).to(device).eval()
        channels, img_size = 3, 32
    
    return model, channels, img_size


def tensor_to_image(t):
    """Convert tensor to displayable image."""
    t = (t.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    t = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (t * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Simple StepDrop Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="all",
                        choices=["ddpm", "ddim", "stepdrop", "all"])
    parser.add_argument("--steps", type=int, default=50, help="DDIM/StepDrop steps")
    parser.add_argument("--output", type=str, default="results/demo.png")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda, mps, cpu). Auto-detected if not specified.")
    args = parser.parse_args()
    
    # Auto-detect device
    device = args.device if args.device else get_best_device()
    print(f"🖥️  Device: {get_device_info(device)}")
    
    # Auto-find checkpoint
    if args.checkpoint is None:
        for p in ["checkpoints/model.pt", "cifar10_64ch_50ep.pt"]:
            if Path(p).exists():
                args.checkpoint = p
                break
    
    model, channels, img_size = load_model(args.checkpoint, device)
    shape = (1, channels, img_size, img_size)
    
    # Set seed
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)
    # MPS uses the global torch seed
    
    # Store initial noise for display
    initial_noise = torch.randn(shape, device=device)
    
    print(f"\n🎲 Seed: {args.seed}")
    print(f"📐 Shape: {shape}")
    print("\n" + "="*50)
    
    results = {}
    
    # Generate with different methods
    methods_to_run = ["ddim", "stepdrop"] if args.method == "all" else [args.method]
    
    if "ddpm" in methods_to_run:
        print("\n🔄 DDPM (1000 steps)... this may take a while")
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed(args.seed)
        sampler = DDPMSampler(num_timesteps=1000)
        with torch.no_grad():
            x = sampler.sample(model, shape, device=device, show_progress=True)
        results["DDPM\n(1000 steps)"] = tensor_to_image(x)
    
    if "ddim" in methods_to_run:
        print(f"\n🔄 DDIM ({args.steps} steps)...")
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed(args.seed)
        sampler = DDIMSampler(num_timesteps=1000, num_inference_steps=args.steps)
        with torch.no_grad():
            x = sampler.sample(model, shape, device=device, show_progress=True)
        results[f"DDIM\n({args.steps} steps)"] = tensor_to_image(x)
    
    if "stepdrop" in methods_to_run:
        print(f"\n🔄 StepDrop (target {args.steps} NFE)...")
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed(args.seed)
        sampler = TargetNFEStepDropSampler(num_timesteps=1000)
        with torch.no_grad():
            x, stats = sampler.sample(
                model, shape, device=device,
                target_nfe=args.steps,
                selection_strategy="importance",
                show_progress=True
            )
        nfe = stats['steps_taken'] if stats else args.steps
        results[f"StepDrop\n({nfe} NFE)"] = tensor_to_image(x)
    
    # Prepare noise visualization
    noise_viz = initial_noise.squeeze(0).permute(1, 2, 0).cpu().numpy()
    noise_viz = (noise_viz - noise_viz.min()) / (noise_viz.max() - noise_viz.min())
    noise_viz = (noise_viz * 255).astype(np.uint8)
    
    # Plot
    n_cols = len(results) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Noise
    axes[0].imshow(noise_viz)
    axes[0].set_title("Initial Noise\n(Input)", fontsize=12)
    axes[0].axis('off')
    
    # Results
    for i, (title, img) in enumerate(results.items()):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(title, fontsize=12)
        axes[i + 1].axis('off')
    
    # Add arrow
    fig.text(0.5, 0.02, "Noise → Denoise → Image", ha='center', fontsize=14, style='italic')
    
    plt.suptitle(f"StepDrop Demo ({get_device_info(device)})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n💾 Saved: {output_path}")
    
    plt.show()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()