#!/usr/bin/env python3
"""
StepDrop Comprehensive Benchmarking Harness
============================================

Runs rigorous evaluation of diffusion sampling strategies with industry-standard metrics.

Metrics Computed:
- FID (Fréchet Inception Distance) - Lower is better
- IS (Inception Score) - Higher is better
- Throughput (Images/Second) - Higher is better
- NFE (Number of Function Evaluations) - Lower is better

Usage:
    # Quick test with dummy model
    python scripts/benchmark_strategies.py --dummy --samples 100

    # Full evaluation with trained model
    python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 5000

    # Custom output directory
    python scripts/benchmark_strategies.py --checkpoint model.pt --output_dir my_results
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Strategy Configuration
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for a sampling strategy."""
    name: str
    type: str  # "ddpm", "ddim", "stepdrop"
    params: Dict[str, Any] = field(default_factory=dict)
    expected_nfe: Optional[int] = None
    description: str = ""


@dataclass
class StrategyResult:
    """Results from evaluating a single strategy."""
    name: str
    fid: float = -1.0
    is_mean: float = -1.0
    is_std: float = -1.0
    throughput: float = -1.0
    duration: float = -1.0
    nfe: int = -1
    num_samples: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fid": self.fid,
            "is_mean": self.is_mean,
            "is_std": self.is_std,
            "throughput": self.throughput,
            "duration": self.duration,
            "nfe": self.nfe,
            "num_samples": self.num_samples,
            "error": self.error
        }


# =============================================================================
# Default Strategies
# =============================================================================

DEFAULT_STRATEGIES = [
    StrategyConfig(
        name="DDPM_1000",
        type="ddpm",
        params={},
        expected_nfe=1000,
        description="Standard DDPM with 1000 steps"
    ),
    StrategyConfig(
        name="DDIM_50",
        type="ddim",
        params={"num_inference_steps": 50, "eta": 0.0},
        expected_nfe=50,
        description="DDIM with 50 deterministic steps"
    ),
    StrategyConfig(
        name="DDIM_25",
        type="ddim",
        params={"num_inference_steps": 25, "eta": 0.0},
        expected_nfe=25,
        description="DDIM with 25 deterministic steps"
    ),
    StrategyConfig(
        name="StepDrop_0.3",
        type="stepdrop",
        params={"skip_strategy": "linear", "base_skip_prob": 0.3},
        expected_nfe=700,  # ~30% skipped
        description="StepDrop with 30% skip probability"
    ),
]


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def tensor_to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] or [0, 1] to uint8 [0, 255]."""
    if images.min() < 0:
        images = (images + 1) / 2
    images = (images * 255).clamp(0, 255).byte()
    return images


# =============================================================================
# Real Data Preparation
# =============================================================================

def prepare_real_data(
    out_dir: str = "data/fid_real_cifar10",
    num_images: int = 10000,
    data_dir: str = "data"
) -> Path:
    """Save CIFAR-10 real images as PNGs for FID reference."""
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    out_path = ensure_dir(out_dir)
    
    # Check existing
    existing = len(list(out_path.glob("*.png")))
    if existing >= num_images:
        print(f"✓ Real data cache exists: {out_path} ({existing} images)")
        return out_path
    
    print(f"📦 Preparing {num_images} CIFAR-10 real images...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    saved = 0
    pbar = tqdm(total=num_images, desc="Saving real images")
    
    for images, _ in loader:
        for img in images:
            if saved >= num_images:
                break
            save_image(img, out_path / f"{saved:06d}.png", normalize=True, value_range=(-1, 1))
            saved += 1
            pbar.update(1)
        if saved >= num_images:
            break
    
    pbar.close()
    print(f"✓ Saved {saved} real images to {out_path}")
    return out_path


# =============================================================================
# Fake Image Generation
# =============================================================================

@torch.no_grad()
def generate_and_save_fake_images(
    generator_fn: Callable[[int], torch.Tensor],
    out_dir: str,
    num_images: int,
    batch_size: int = 16,
    device: str = "cuda"
) -> Tuple[Path, float, float]:
    """Generate fake images and save them. Returns (path, duration, throughput)."""
    out_path = ensure_dir(out_dir)
    
    # Clear existing
    for f in out_path.glob("*.png"):
        f.unlink()
    
    saved = 0
    pbar = tqdm(total=num_images, desc="Generating samples")
    
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    
    while saved < num_images:
        current_batch = min(batch_size, num_images - saved)
        
        images = generator_fn(current_batch)
        
        # Handle tuple returns (images, stats)
        if isinstance(images, tuple):
            images = images[0]
        
        images = images.cpu()
        
        for img in images:
            if saved >= num_images:
                break
            save_image(img, out_path / f"{saved:06d}.png", normalize=True, value_range=(-1, 1))
            saved += 1
            pbar.update(1)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    duration = time.time() - start_time
    throughput = num_images / duration
    
    pbar.close()
    return out_path, duration, throughput


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_fid_score(real_dir: str, fake_dir: str, device: str = "cuda") -> float:
    """Compute FID between real and fake image directories."""
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        
        fid = calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=64,
            device=device,
            dims=2048
        )
        return fid
    except ImportError:
        print("  ⚠️ pytorch-fid not installed. Run: pip install pytorch-fid")
        return -1.0
    except Exception as e:
        print(f"  ⚠️ FID computation failed: {e}")
        return -1.0


def compute_inception_score(fake_dir: str, device: str = "cuda") -> Tuple[float, float]:
    """Compute Inception Score from fake image directory."""
    try:
        from torchmetrics.image.inception import InceptionScore
        import torchvision.transforms as transforms
        from PIL import Image
        
        inception = InceptionScore(normalize=True).to(device)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        image_files = sorted(Path(fake_dir).glob("*.png"))[:1000]  # Limit for speed
        
        batch_size = 32
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            images = []
            for f in batch_files:
                img = Image.open(f).convert("RGB")
                img = transform(img)
                images.append(img)
            
            batch = torch.stack(images).to(device)
            batch_uint8 = (batch * 255).clamp(0, 255).byte()
            inception.update(batch_uint8)
        
        mean, std = inception.compute()
        return mean.item(), std.item()
        
    except ImportError:
        print("  ⚠️ torchmetrics not installed. Run: pip install torchmetrics[image]")
        return -1.0, -1.0
    except Exception as e:
        print(f"  ⚠️ IS computation failed: {e}")
        return -1.0, -1.0


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: Optional[str], device: str, dummy: bool = False) -> nn.Module:
    """Load model from checkpoint or create dummy."""
    if dummy:
        print("🎭 Using dummy model for testing...")
        
        class DummyUNet(nn.Module):
            def __init__(self, channels=3):
                super().__init__()
                self.conv = nn.Conv2d(channels, channels, 3, padding=1)
            
            def forward(self, x, t):
                return self.conv(x) * 0.1
        
        return DummyUNet(channels=3).to(device)
    
    if checkpoint_path is None:
        raise ValueError("Must provide --checkpoint or use --dummy")
    
    print(f"📦 Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config", {})
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        from src.modules import TinyUNet
        model = TinyUNet(
            img_size=config.get("img_size", 32),
            channels=config.get("channels", 3),
            base_channels=config.get("base_channels", 64)
        ).to(device)
        
        model.load_state_dict(state_dict)
    else:
        model = checkpoint
    
    model.eval()
    return model


# =============================================================================
# Sampler Factory
# =============================================================================

def create_sampler(strategy: StrategyConfig, num_timesteps: int = 1000):
    """Create sampler based on strategy type."""
    try:
        if strategy.type == "ddpm":
            from scripts.ddpm_sampler import DDPMSampler
            return DDPMSampler(num_timesteps=num_timesteps)
        elif strategy.type == "ddim":
            from scripts.ddim_sampler import DDIMSampler
            return DDIMSampler(num_timesteps=num_timesteps)
        elif strategy.type == "stepdrop":
            from scripts.step_skipper import StochasticStepSkipScheduler
            return StochasticStepSkipScheduler(num_timesteps=num_timesteps)
        else:
            raise ValueError(f"Unknown strategy type: {strategy.type}")
    except ImportError as e:
        print(f"  ⚠️ Could not import sampler: {e}")
        return None


def create_generator(
    model: nn.Module,
    strategy: StrategyConfig,
    image_shape: Tuple[int, ...],
    device: str,
    num_timesteps: int = 1000
) -> Callable[[int], torch.Tensor]:
    """Create generator function for a strategy."""
    sampler = create_sampler(strategy, num_timesteps)
    
    if sampler is None:
        # Fallback: simple noise
        def fallback_generator(n: int) -> torch.Tensor:
            return torch.randn(n, *image_shape, device=device)
        return fallback_generator
    
    def generator(n_samples: int) -> torch.Tensor:
        shape = (n_samples, *image_shape)
        
        if strategy.type == "ddpm":
            return sampler.sample(model, shape, device=device)
        elif strategy.type == "ddim":
            steps = strategy.params.get("num_inference_steps", 50)
            eta = strategy.params.get("eta", 0.0)
            return sampler.sample(model, shape, num_inference_steps=steps, eta=eta, device=device)
        elif strategy.type == "stepdrop":
            return sampler.sample(model, shape, device=device, **strategy.params)
        else:
            return torch.randn(shape, device=device)
    
    return generator


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Runs comprehensive benchmarks on diffusion sampling strategies."""
    
    def __init__(
        self,
        output_dir: str = "results",
        device: str = "cuda",
        num_timesteps: int = 1000,
        image_shape: Tuple[int, int, int] = (3, 32, 32)
    ):
        self.device = device
        self.num_timesteps = num_timesteps
        self.image_shape = image_shape
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, StrategyResult] = {}
        
        print(f"📁 Results will be saved to: {self.output_dir}")
    
    def run_strategy(
        self,
        strategy: StrategyConfig,
        model: nn.Module,
        num_samples: int,
        batch_size: int,
        real_data_dir: str
    ) -> StrategyResult:
        """Run evaluation for a single strategy."""
        print(f"\n{'='*60}")
        print(f"🚀 Strategy: {strategy.name}")
        print(f"   {strategy.description}")
        print(f"{'='*60}")
        
        result = StrategyResult(name=strategy.name, num_samples=num_samples)
        
        try:
            # Create output directory for this strategy
            strategy_dir = self.output_dir / strategy.name
            samples_dir = strategy_dir / "samples"
            
            # Create generator
            generator = create_generator(
                model, strategy, self.image_shape, self.device, self.num_timesteps
            )
            
            # Generate and save images
            print(f"   Generating {num_samples} samples...")
            fake_dir, duration, throughput = generate_and_save_fake_images(
                generator,
                str(samples_dir),
                num_samples,
                batch_size,
                self.device
            )
            
            result.duration = duration
            result.throughput = throughput
            result.nfe = strategy.expected_nfe or -1
            
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   ⚡ Throughput: {throughput:.2f} img/s")
            
            # Compute FID
            print("   📊 Computing FID...")
            result.fid = compute_fid_score(real_data_dir, str(fake_dir), self.device)
            if result.fid > 0:
                print(f"   📈 FID: {result.fid:.2f}")
            
            # Compute IS
            print("   📊 Computing Inception Score...")
            result.is_mean, result.is_std = compute_inception_score(str(fake_dir), self.device)
            if result.is_mean > 0:
                print(f"   📈 IS: {result.is_mean:.2f} ± {result.is_std:.2f}")
            
        except Exception as e:
            result.error = str(e)
            print(f"   ❌ Error: {e}")
        
        self.results[strategy.name] = result
        return result
    
    def run_all(
        self,
        strategies: List[StrategyConfig],
        model: nn.Module,
        num_samples: int,
        batch_size: int,
        real_data_dir: str
    ):
        """Run all strategies."""
        for strategy in strategies:
            self.run_strategy(strategy, model, num_samples, batch_size, real_data_dir)
        
        self.save_report()
        self.print_summary()
    
    def save_report(self):
        """Save results to JSON."""
        report = {
            name: result.to_dict()
            for name, result in self.results.items()
        }
        
        report_path = self.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📝 Report saved to: {report_path}")
    
    def print_summary(self):
        """Print summary table."""
        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Strategy':<25} {'FID':>10} {'IS':>12} {'Throughput':>15} {'NFE':>8}")
        print("-" * 80)
        
        for name, result in self.results.items():
            fid_str = f"{result.fid:.2f}" if result.fid > 0 else "N/A"
            is_str = f"{result.is_mean:.2f}±{result.is_std:.2f}" if result.is_mean > 0 else "N/A"
            tp_str = f"{result.throughput:.2f} img/s" if result.throughput > 0 else "N/A"
            nfe_str = str(result.nfe) if result.nfe > 0 else "N/A"
            
            print(f"{name:<25} {fid_str:>10} {is_str:>12} {tp_str:>15} {nfe_str:>8}")
        
        print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="StepDrop Benchmarking Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  1. Quick test with dummy model:
     python scripts/benchmark_strategies.py --dummy --samples 50

  2. Full evaluation with trained model:
     python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 1000

  3. Custom batch size:
     python scripts/benchmark_strategies.py --checkpoint model.pt --batch_size 32

OUTPUT:
  Results saved to: results/YYYY-MM-DD_HH-MM-SS/
    - report.json (metrics for all strategies)
    - <strategy>/samples/*.png (generated images)
"""
    )
    
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy model (for testing)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples per strategy")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--real_data_dir", type=str, default="data/fid_real_cifar10",
                        help="Directory with real images for FID")
    parser.add_argument("--strategies", type=str, default="all",
                        help="Comma-separated strategy names or 'all'")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Prepare real data
    num_real = 100 if args.dummy else 5000
    real_data_dir = prepare_real_data(
        out_dir=args.real_data_dir,
        num_images=num_real
    )
    
    # Load model
    model = load_model(args.checkpoint, device, dummy=args.dummy)
    
    # Select strategies
    if args.strategies == "all":
        strategies = DEFAULT_STRATEGIES
    else:
        names = [s.strip() for s in args.strategies.split(",")]
        strategies = [s for s in DEFAULT_STRATEGIES if s.name in names]
    
    # For dummy/quick runs, use fewer strategies
    if args.dummy and len(strategies) > 2:
        strategies = strategies[:2]
        print(f"🎭 Dummy mode: Using only {len(strategies)} strategies")
    
    # Run benchmark
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        device=device
    )
    
    runner.run_all(
        strategies=strategies,
        model=model,
        num_samples=args.samples,
        batch_size=args.batch_size,
        real_data_dir=str(real_data_dir)
    )
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
