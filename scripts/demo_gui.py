#!/usr/bin/env python3
"""
Interactive StepDrop Demo
=========================

A simple GUI to generate images from random noise using different sampling methods.
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

Usage:
    python scripts/demo_gui.py
    python scripts/demo_gui.py --checkpoint checkpoints/model.pt
    python scripts/demo_gui.py --share  # Create public link

Requirements:
    pip install gradio
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules import TinyUNet
from src.sampler import DDPMSampler, DDIMSampler, StepDropSampler, AdaptiveStepDropSampler, TargetNFEStepDropSampler

# Check for gradio
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("⚠️  Gradio not installed. Install with: pip install gradio")


# =============================================================================
# Device Selection
# =============================================================================

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


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📦 Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            img_size = config.get('img_size', 32)
            channels = config.get('channels', 3)
            base_channels = config.get('base_channels', 64)
            model_state = checkpoint['model_state_dict']
        else:
            # Assume CIFAR-10 defaults
            img_size, channels, base_channels = 32, 3, 64
            model_state = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model_state_dict', checkpoint)
        
        model = TinyUNet(img_size=img_size, channels=channels, base_channels=base_channels)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        return model, {'img_size': img_size, 'channels': channels, 'base_channels': base_channels}
    else:
        print("⚠️  No checkpoint provided. Using dummy model (random outputs).")
        # Create a dummy model for demonstration
        model = TinyUNet(img_size=32, channels=3, base_channels=32)
        model.to(device)
        model.eval()
        return model, {'img_size': 32, 'channels': 3, 'base_channels': 32}


# =============================================================================
# Sampling Functions
# =============================================================================

def generate_image(
    method: str,
    seed: int,
    ddim_steps: int,
    skip_prob: float,
    skip_strategy: str,
    target_nfe: int,
    model,
    config: dict,
    device: str
):
    """Generate a single image with the specified method."""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    # Note: MPS doesn't have a separate manual_seed function
    np.random.seed(seed)
    
    img_size = config['img_size']
    channels = config['channels']
    shape = (1, channels, img_size, img_size)
    
    with torch.no_grad():
        if method == "DDPM (1000 steps)":
            sampler = DDPMSampler(num_timesteps=1000)
            samples = sampler.sample(model, shape, device=device, show_progress=False)
            nfe = 1000
            
        elif method == "DDIM":
            sampler = DDIMSampler(num_timesteps=1000, num_inference_steps=ddim_steps)
            samples = sampler.sample(model, shape, device=device, show_progress=False)
            nfe = ddim_steps
            
        elif method == "StepDrop (Probability)":
            sampler = StepDropSampler(num_timesteps=1000)
            samples, stats = sampler.sample(
                model, shape, device=device,
                skip_prob=skip_prob,
                skip_strategy=skip_strategy,
                show_progress=False
            )
            nfe = stats.steps_taken if stats else "N/A"
            
        elif method == "StepDrop (Target NFE)":
            sampler = TargetNFEStepDropSampler(num_timesteps=1000)
            samples, stats = sampler.sample(
                model, shape, device=device,
                target_nfe=target_nfe,
                selection_strategy="importance",
                show_progress=False
            )
            nfe = stats['steps_taken'] if stats else target_nfe
            
        elif method == "Adaptive StepDrop":
            sampler = AdaptiveStepDropSampler(num_timesteps=1000)
            samples, stats = sampler.sample(
                model, shape, device=device,
                base_skip_prob=skip_prob,
                show_progress=False
            )
            nfe = stats['steps_taken'] if stats else "N/A"
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Convert to image format [0, 255]
    samples = (samples.clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
    samples = samples.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
    samples = (samples * 255).astype(np.uint8)
    
    # Handle grayscale
    if channels == 1:
        samples = samples.squeeze(-1)

    # Upscale to fill the display area (e.g., 768x768)
    from PIL import Image
    target_size = 768  # or 512, or whatever your gr.Image size is
    if samples.ndim == 2:  # grayscale
        img = Image.fromarray(samples, mode="L")
    else:
        img = Image.fromarray(samples)
    img = img.resize((target_size, target_size), resample=Image.NEAREST)
    samples = np.array(img)

    return samples, f"NFE: {nfe}"


def generate_noise_preview(seed: int, img_size: int = 32, channels: int = 3):
    """Generate a preview of the initial noise."""
    torch.manual_seed(seed)
    noise = torch.randn(1, channels, img_size, img_size)

    # Normalize for visualization
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = noise.squeeze(0).permute(1, 2, 0).numpy()
    noise = (noise * 255).astype(np.uint8)

    if channels == 1:
        noise = noise.squeeze(-1)

    # Upscale for display
    from PIL import Image
    target_size = 768
    if noise.ndim == 2:
        img = Image.fromarray(noise, mode="L")
    else:
        img = Image.fromarray(noise)
    img = img.resize((target_size, target_size), resample=Image.NEAREST)
    noise = np.array(img)

    return noise


# =============================================================================
# Gradio Interface
# =============================================================================

def create_demo(checkpoint_path: str, device: str):
    """Create the Gradio demo interface."""
    
    # Load model once
    model, config = load_model(checkpoint_path, device)
    device_info = get_device_info(device)
    
    def generate_wrapper(method, seed, ddim_steps, skip_prob, skip_strategy, target_nfe):
        """Wrapper for Gradio."""
        image, info = generate_image(
            method, seed, ddim_steps, skip_prob, skip_strategy, target_nfe,
            model, config, device
        )
        noise = generate_noise_preview(seed, config['img_size'], config['channels'])
        return noise, image, info
    
    def randomize_seed():
        """Generate a random seed."""
        return np.random.randint(0, 2**31)
    
    # Build interface
    with gr.Blocks(title="StepDrop Demo") as demo:
        gr.Markdown(f"""
        # 🎨 StepDrop: Stochastic Step Skipping Demo
        
        Generate images from **random noise** using different diffusion sampling methods.
        
        **Authors:**
        - Wanghley Soares Martins
        - Nicolas Vasilescu
        - Logan Chu
        
        **Device:** {device_info}
        
        **How it works:**
        1. Start with pure Gaussian noise (left image)
        2. The model iteratively denoises it
        3. Final generated image appears (right)
        
        Try different methods and see how they affect quality and speed!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                
                method = gr.Dropdown(
                    choices=[
                        "DDPM (1000 steps)",
                        "DDIM",
                        "StepDrop (Probability)",
                        "StepDrop (Target NFE)",
                        "Adaptive StepDrop"
                    ],
                    value="DDIM",
                    label="Sampling Method"
                )
                
                with gr.Row():
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    random_btn = gr.Button("🎲 Random", size="sm")
                
                gr.Markdown("---")
                gr.Markdown("#### DDIM Settings")
                ddim_steps = gr.Slider(10, 200, value=50, step=5, label="DDIM Steps")
                
                gr.Markdown("#### StepDrop Settings")
                skip_prob = gr.Slider(0.0, 0.8, value=0.3, step=0.05, label="Skip Probability")
                skip_strategy = gr.Dropdown(
                    choices=["constant", "linear", "cosine_sq", "quadratic", "early_skip", "late_skip"],
                    value="linear",
                    label="Skip Strategy"
                )
                target_nfe = gr.Slider(10, 200, value=50, step=5, label="Target NFE")
                
                generate_btn = gr.Button("🚀 Generate!", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📸 Results")
                with gr.Row():
                    noise_output = gr.Image(
                        label="Initial Noise (Input)",
                        width=512,
                        height=512,
                        show_label=True
                    )
                    image_output = gr.Image(
                        label="Generated Image (Output)",
                        width=512,
                        height=512,
                        show_label=True
                    )
                info_output = gr.Textbox(label="Info", interactive=False)
                
                gr.Markdown("""
                ---
                ### 📖 Method Comparison
                
                | Method | Steps | Speed | Quality |
                |:-------|:------|:------|:--------|
                | DDPM | 1000 | 🐢 Slow | ⭐⭐⭐ Best |
                | DDIM | 10-200 | 🚀 Fast | ⭐⭐ Good |
                | StepDrop | Variable | 🚀 Fast | ⭐⭐ Good |
                | Adaptive | Dynamic | 🚀 Fast | ⭐⭐ Good |
                """)
        
        # Event handlers
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[method, seed, ddim_steps, skip_prob, skip_strategy, target_nfe],
            outputs=[noise_output, image_output, info_output]
        )
        
        random_btn.click(
            fn=randomize_seed,
            outputs=[seed]
        )
        
        # Also generate on seed change for quick exploration
        seed.change(
            fn=generate_wrapper,
            inputs=[method, seed, ddim_steps, skip_prob, skip_strategy, target_nfe],
            outputs=[noise_output, image_output, info_output]
        )
    
    return demo


# =============================================================================
# Simple CLI Demo (fallback without Gradio)
# =============================================================================

def cli_demo(checkpoint_path: str, device: str):
    """Simple CLI demo if Gradio is not available."""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("🎨 StepDrop CLI Demo")
    print("="*60)
    print(f"Device: {get_device_info(device)}")
    
    model, config = load_model(checkpoint_path, device)
    
    methods = ["DDIM", "StepDrop (Probability)", "StepDrop (Target NFE)"]
    seed = 42
    
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(4 * (len(methods) + 1), 4))
    
    # Show initial noise
    noise = generate_noise_preview(seed, config['img_size'], config['channels'])
    axes[0].imshow(noise)
    axes[0].set_title("Initial Noise")
    axes[0].axis('off')
    
    # Generate with each method
    for i, method in enumerate(methods):
        print(f"\n🔄 Generating with {method}...")
        image, info = generate_image(
            method, seed, 
            ddim_steps=50, 
            skip_prob=0.3, 
            skip_strategy="linear",
            target_nfe=50,
            model=model, 
            config=config, 
            device=device
        )
        axes[i + 1].imshow(image)
        axes[i + 1].set_title(f"{method}\n{info}")
        axes[i + 1].axis('off')
        print(f"   ✅ {info}")
    
    plt.tight_layout()
    
    # Save and show
    output_path = PROJECT_ROOT / "results" / "demo_output.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved to {output_path}")
    
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive StepDrop Demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port for Gradio server")
    parser.add_argument("--cli", action="store_true",
                        help="Use CLI demo instead of GUI")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = get_best_device()
    
    print(f"🖥️  Device: {get_device_info(args.device)}")
    
    # Auto-detect checkpoint
    if args.checkpoint is None:
        # Look for common checkpoint locations
        candidates = [
            "checkpoints/model.pt",
            "checkpoints/cifar10_model.pt",
            "cifar10_64ch_50ep.pt",
        ]
        for c in candidates:
            if Path(c).exists():
                args.checkpoint = c
                print(f"📦 Auto-detected checkpoint: {c}")
                break
    
    if args.cli or not HAS_GRADIO:
        if not HAS_GRADIO:
            print("⚠️  Gradio not available, using CLI demo.")
            print("   Install with: pip install gradio")
        cli_demo(args.checkpoint, args.device)
    else:
        demo = create_demo(args.checkpoint, args.device)
        demo.launch(
            share=args.share,
            server_port=args.port,
            server_name="0.0.0.0",
            theme=gr.themes.Soft()
        )


if __name__ == "__main__":
    main()