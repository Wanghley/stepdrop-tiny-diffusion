#!/usr/bin/env python3
"""
StepDrop Comprehensive Benchmarking Harness
============================================

Runs rigorous evaluation of diffusion sampling strategies with industry-standard metrics.

Metrics Computed:
- Distribution: FID, KID, IS, Precision, Recall, Density, Coverage
- Perceptual: LPIPS, SSIM, PSNR, MS-SSIM
- Diversity: Vendi Score, Intra-LPIPS
- Efficiency: FLOPs, Throughput, Memory, NFE

Usage:
    # Quick test with dummy model
    python scripts/benchmark_strategies.py --dummy --samples 100

    # Full evaluation with trained model
    python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 5000

    # Comprehensive evaluation with all metrics
    python scripts/benchmark_strategies.py --checkpoint model.pt --samples 5000 --full-metrics
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

# Import from project
from src.modules import TinyUNet
from src.scheduler import NoiseScheduler
from src.sampler import DDPMSampler, DDIMSampler, StepDropSampler, AdaptiveStepDropSampler, TargetNFEStepDropSampler
from src.eval.metrics_utils import (
    DiffusionEvaluator,
    EvaluationReport,
    MetricResult,
    save_cifar10_real_subset,
    save_fake_images,
    load_images_from_dir,
    compute_fid,
    compute_kid,
    compute_inception_score,
    compute_precision_recall,
    compute_density_coverage,
    compute_lpips,
    compute_ssim,
    compute_psnr,
    compute_vendi_score,
    compute_intra_lpips,
    compute_flops,
    compute_throughput,
    compute_memory_usage,
    InceptionFeatureExtractor,
    ensure_dir,
)


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
    """Comprehensive results from evaluating a single strategy."""
    name: str
    
    # Distribution metrics
    fid: float = -1.0
    kid: float = -1.0
    is_mean: float = -1.0
    is_std: float = -1.0
    precision: float = -1.0
    recall: float = -1.0
    density: float = -1.0
    coverage: float = -1.0
    
    # Perceptual metrics
    lpips: float = -1.0
    ssim: float = -1.0
    psnr: float = -1.0
    ms_ssim: float = -1.0
    
    # Diversity metrics
    vendi_score: float = -1.0
    intra_lpips: float = -1.0
    
    # Efficiency metrics
    throughput: float = -1.0
    duration: float = -1.0
    nfe: int = -1
    flops: float = -1.0
    memory_gb: float = -1.0
    
    # Metadata
    num_samples: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            # Distribution
            "fid": self.fid,
            "kid": self.kid,
            "is_mean": self.is_mean,
            "is_std": self.is_std,
            "precision": self.precision,
            "recall": self.recall,
            "density": self.density,
            "coverage": self.coverage,
            # Perceptual
            "lpips": self.lpips,
            "ssim": self.ssim,
            "psnr": self.psnr,
            "ms_ssim": self.ms_ssim,
            # Diversity
            "vendi_score": self.vendi_score,
            "intra_lpips": self.intra_lpips,
            # Efficiency
            "throughput": self.throughput,
            "duration": self.duration,
            "nfe": self.nfe,
            "flops": self.flops,
            "memory_gb": self.memory_gb,
            # Meta
            "num_samples": self.num_samples,
            "error": self.error
        }


# =============================================================================
# Default Strategies
# =============================================================================

DEFAULT_STRATEGIES = [
    # ==========================================================================
    # Baseline Methods
    # ==========================================================================
    StrategyConfig(
        name="DDPM_1000",
        type="ddpm",
        params={},
        expected_nfe=1000,
        description="Standard DDPM with 1000 steps (baseline)"
    ),
    StrategyConfig(
        name="DDIM_100",
        type="ddim",
        params={"num_inference_steps": 100, "eta": 0.0},
        expected_nfe=100,
        description="DDIM with 100 deterministic steps"
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
        name="DDIM_10",
        type="ddim",
        params={"num_inference_steps": 10, "eta": 0.0},
        expected_nfe=10,
        description="DDIM with 10 deterministic steps (fast)"
    ),
    
    # ==========================================================================
    # StepDrop: Linear Schedule (parabolic skip probability)
    # p(t) = base_prob * 4 * t * (1-t) - peaks at middle
    # ==========================================================================
    StrategyConfig(
        name="StepDrop_Linear_0.3",
        type="stepdrop",
        params={"skip_strategy": "linear", "base_skip_prob": 0.3},
        expected_nfe=700,
        description="StepDrop Linear schedule, 30% base skip"
    ),
    StrategyConfig(
        name="StepDrop_Linear_0.5",
        type="stepdrop",
        params={"skip_strategy": "linear", "base_skip_prob": 0.5},
        expected_nfe=500,
        description="StepDrop Linear schedule, 50% base skip"
    ),
    
    # ==========================================================================
    # StepDrop: Cosine² Schedule (smooth transitions)
    # p(t) = base_prob * sin²(πt) - smoother than linear
    # ==========================================================================
    StrategyConfig(
        name="StepDrop_CosineSq_0.3",
        type="stepdrop",
        params={"skip_strategy": "cosine_sq", "base_skip_prob": 0.3},
        expected_nfe=700,
        description="StepDrop Cosine² schedule, 30% base skip"
    ),
    StrategyConfig(
        name="StepDrop_CosineSq_0.5",
        type="stepdrop",
        params={"skip_strategy": "cosine_sq", "base_skip_prob": 0.5},
        expected_nfe=500,
        description="StepDrop Cosine² schedule, 50% base skip"
    ),
    
    # ==========================================================================
    # StepDrop: Quadratic Schedule (sharper peak in middle)
    # p(t) = base_prob * 16 * t² * (1-t)² - more aggressive middle skipping
    # ==========================================================================
    StrategyConfig(
        name="StepDrop_Quadratic_0.3",
        type="stepdrop",
        params={"skip_strategy": "quadratic", "base_skip_prob": 0.3},
        expected_nfe=750,
        description="StepDrop Quadratic schedule, 30% base skip"
    ),
    StrategyConfig(
        name="StepDrop_Quadratic_0.5",
        type="stepdrop",
        params={"skip_strategy": "quadratic", "base_skip_prob": 0.5},
        expected_nfe=550,
        description="StepDrop Quadratic schedule, 50% base skip"
    ),
    
    # ==========================================================================
    # StepDrop: Adaptive (error-based dynamic skipping)
    # ==========================================================================
    StrategyConfig(
        name="StepDrop_Adaptive",
        type="stepdrop_adaptive",
        params={"base_skip_prob": 0.3},
        expected_nfe=700,
        description="Adaptive StepDrop with error-based feedback"
    ),
    
    # ==========================================================================
    # TargetNFE StepDrop: Compete directly with DDIM at same NFE
    # ==========================================================================
    StrategyConfig(
        name="StepDrop_Target50_Uniform",
        type="stepdrop_target",
        params={"target_nfe": 50, "selection_strategy": "uniform"},
        expected_nfe=50,
        description="StepDrop targeting 50 NFE with uniform selection"
    ),
    StrategyConfig(
        name="StepDrop_Target50_Importance",
        type="stepdrop_target",
        params={"target_nfe": 50, "selection_strategy": "importance"},
        expected_nfe=50,
        description="StepDrop targeting 50 NFE with importance-weighted selection"
    ),
    StrategyConfig(
        name="StepDrop_Target50_Stochastic",
        type="stepdrop_target",
        params={"target_nfe": 50, "selection_strategy": "stochastic"},
        expected_nfe=50,
        description="StepDrop targeting 50 NFE with stochastic selection"
    ),
    StrategyConfig(
        name="StepDrop_Target25_Importance",
        type="stepdrop_target",
        params={"target_nfe": 25, "selection_strategy": "importance"},
        expected_nfe=25,
        description="StepDrop targeting 25 NFE with importance-weighted selection"
    ),
]


# =============================================================================
# Utility Functions
# =============================================================================

def tensor_to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] or [0, 1] to uint8 [0, 255]."""
    if images.min() < 0:
        images = (images + 1) / 2
    images = (images * 255).clamp(0, 255).to(torch.uint8)
    return images


# =============================================================================
# Real Data Preparation
# =============================================================================

def prepare_real_data(
    out_dir: str = "data/fid_real_cifar10",
    num_images: int = 10000,
    dataset: str = "cifar10"
) -> str:
    """Prepare real data for FID computation."""
    out_path = Path(out_dir)
    
    if out_path.exists() and len(list(out_path.glob("*.png"))) >= num_images:
        print(f"✅ Real data cache exists: {out_dir} ({num_images} images)")
        return str(out_path)
    
    print(f"📥 Preparing real data cache ({num_images} images)...")
    
    if dataset == "mnist":
        # Import MNIST dataset
        from torchvision import datasets, transforms
        from torchvision.utils import save_image
        
        out_path.mkdir(parents=True, exist_ok=True)
        
        transform = transforms.Compose([
            transforms.Resize(32),  # Resize to 32x32 for consistency
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        for i in range(min(num_images, len(mnist))):
            img, _ = mnist[i]
            save_image(img, out_path / f"{i:05d}.png", normalize=True, value_range=(-1, 1))
        
        print(f"✅ Saved {num_images} MNIST images to {out_dir}")
        return str(out_path)
    else:
        return str(save_cifar10_real_subset(num_images=num_images, out_dir=out_dir))


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: Optional[str], device: str, dummy: bool = False) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model from checkpoint or create dummy. Returns model and config dict."""
    config = {
        'img_size': 32,
        'channels': 3,
        'base_channels': 64
    }
    
    if dummy or checkpoint_path is None:
        print("🎭 Using dummy model for testing")
        model = TinyUNet(img_size=config['img_size'], channels=config['channels'], base_channels=config['base_channels'])
    else:
        print(f"📦 Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            config['img_size'] = ckpt_config.get('img_size', 32)
            config['channels'] = ckpt_config.get('channels', 3)
            config['base_channels'] = ckpt_config.get('base_channels', 64)
            model = TinyUNet(
                img_size=config['img_size'],
                channels=config['channels'],
                base_channels=config['base_channels']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Try to infer channels from the first conv layer weights
            state_dict = checkpoint['model_state_dict']
            if 'init_conv.weight' in state_dict:
                config['channels'] = state_dict['init_conv.weight'].shape[1]
            model = TinyUNet(
                img_size=config['img_size'],
                channels=config['channels'],
                base_channels=config['base_channels']
            )
            model.load_state_dict(state_dict)
        else:
            # Assume it's a raw state dict
            if 'init_conv.weight' in checkpoint:
                config['channels'] = checkpoint['init_conv.weight'].shape[1]
            model = TinyUNet(
                img_size=config['img_size'],
                channels=config['channels'],
                base_channels=config['base_channels']
            )
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"   Model config: img_size={config['img_size']}, channels={config['channels']}, base_channels={config['base_channels']}")
    
    return model, config


# =============================================================================
# Sampler Factory
# =============================================================================

def create_sampler(strategy: StrategyConfig, num_timesteps: int = 1000):
    """Create sampler based on strategy type."""
    if strategy.type == "ddpm":
        return DDPMSampler(num_timesteps=num_timesteps)
    elif strategy.type == "ddim":
        return DDIMSampler(
            num_timesteps=num_timesteps,
            num_inference_steps=strategy.params.get("num_inference_steps", 50),
            eta=strategy.params.get("eta", 0.0)
        )
    elif strategy.type == "stepdrop":
        return StepDropSampler(num_timesteps=num_timesteps)
    elif strategy.type == "stepdrop_adaptive":
        return AdaptiveStepDropSampler(num_timesteps=num_timesteps)
    elif strategy.type == "stepdrop_target":
        return TargetNFEStepDropSampler(num_timesteps=num_timesteps)
    else:
        raise ValueError(f"Unknown strategy type: {strategy.type}")


def create_generator(
    model: nn.Module,
    strategy: StrategyConfig,
    image_shape: Tuple[int, ...],
    device: str,
    num_timesteps: int = 1000
) -> Callable[[int], torch.Tensor]:
    """Create generator function for a strategy."""
    sampler = create_sampler(strategy, num_timesteps)
    
    def generator(num_samples: int) -> torch.Tensor:
        shape = (num_samples, *image_shape)
        
        if strategy.type == "ddpm":
            samples = sampler.sample(model, shape, device=device, show_progress=False)
        elif strategy.type == "ddim":
            samples = sampler.sample(model, shape, device=device, show_progress=False)
        elif strategy.type == "stepdrop":
            samples, _ = sampler.sample(
                model, shape, device=device,
                skip_strategy=strategy.params.get("skip_strategy", "linear"),
                skip_prob=strategy.params.get("base_skip_prob", 0.3),
                return_stats=True,
                show_progress=False
            )
        elif strategy.type == "stepdrop_adaptive":
            samples, _ = sampler.sample(
                model, shape, device=device,
                base_skip_prob=strategy.params.get("base_skip_prob", 0.2),
                return_stats=True,
                show_progress=False
            )
        elif strategy.type == "stepdrop_target":
            samples, _ = sampler.sample(
                model, shape, device=device,
                target_nfe=strategy.params.get("target_nfe", 50),
                selection_strategy=strategy.params.get("selection_strategy", "uniform"),
                eta=strategy.params.get("eta", 0.0),
                return_stats=True,
                show_progress=False
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy.type}")
        
        # Normalize to [0, 1] for metrics (most metrics expect this range)
        samples = (samples.clamp(-1, 1) + 1) / 2
        return samples.clamp(0, 1)
    
    return generator


def generate_and_save_fake_images(
    generator: Callable,
    output_dir: str,
    num_images: int,
    batch_size: int,
    device: str
) -> Tuple[Path, float, float]:
    """Generate fake images and save to directory."""
    output_path = ensure_dir(output_dir)
    
    start_time = time.time()
    num_generated = 0
    
    with torch.no_grad():
        while num_generated < num_images:
            current_batch = min(batch_size, num_images - num_generated)
            images = generator(current_batch)
            
            for i, img in enumerate(images):
                save_image(img, output_path / f"{num_generated + i:05d}.png")
            
            num_generated += current_batch
    
    duration = time.time() - start_time
    throughput = num_images / duration if duration > 0 else 0
    
    return output_path, duration, throughput


# =============================================================================
# Comprehensive Benchmark Runner
# =============================================================================

class ComprehensiveBenchmarkRunner:
    """Runs comprehensive benchmarks on diffusion sampling strategies."""
    
    def __init__(
        self,
        output_dir: str = "results",
        device: str = "cuda",
        num_timesteps: int = 1000,
        image_shape: Tuple[int, int, int] = (3, 32, 32),
        compute_full_metrics: bool = False
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = ensure_dir(f"{output_dir}/{timestamp}")
        self.device = device
        self.num_timesteps = num_timesteps
        self.image_shape = image_shape
        self.compute_full_metrics = compute_full_metrics
        self.results: Dict[str, StrategyResult] = {}
        self.feature_extractor = None
        self.real_features = None
    
    def _get_feature_extractor(self) -> InceptionFeatureExtractor:
        """Lazy load feature extractor."""
        if self.feature_extractor is None:
            self.feature_extractor = InceptionFeatureExtractor(self.device)
        return self.feature_extractor
    
    def _get_real_features(self, real_images: torch.Tensor) -> torch.Tensor:
        """Cache real features for efficiency."""
        if self.real_features is None:
            extractor = self._get_feature_extractor()
            self.real_features = extractor.extract(real_images)
        return self.real_features
    
    def run_strategy(
        self,
        strategy: StrategyConfig,
        model: nn.Module,
        num_samples: int,
        batch_size: int,
        real_data_dir: str,
        real_images: Optional[torch.Tensor] = None
    ) -> StrategyResult:
        """Run comprehensive evaluation for a single strategy."""
        result = StrategyResult(name=strategy.name)
        result.description = strategy.description
        result.nfe = strategy.expected_nfe
        result.num_samples = num_samples
        
        print(f"\n{'='*60}")
        print(f"📊 Evaluating: {strategy.name}")
        print(f"   {strategy.description}")
        print(f"   Expected NFE: {strategy.expected_nfe}")
        print(f"{'='*60}")
        
        try:
            # Create generator
            generator = create_generator(
                model, strategy, self.image_shape, self.device, self.num_timesteps
            )
            
            # Generate and save images
            print(f"\n   🎨 Generating {num_samples} samples...")
            fake_dir = str(self.output_dir / strategy.name / "samples")
            fake_dir, duration, throughput = generate_and_save_fake_images(
                generator, fake_dir, num_samples, batch_size, self.device
            )
            
            result.duration = duration
            result.throughput = throughput
            print(f"      ✅ Generated in {duration:.1f}s ({throughput:.1f} img/s)")
            
            # Load images for metrics
            print(f"\n   📈 Computing Metrics...")
            fake_images = load_images_from_dir(str(fake_dir), max_images=num_samples, normalize=False)
            
            # Ensure fake images are in [0, 1] range
            if fake_images.min() < 0:
                fake_images = (fake_images + 1) / 2
            fake_images = fake_images.clamp(0, 1)
            
            # Load real images if not provided
            if real_images is None:
                real_images = load_images_from_dir(real_data_dir, max_images=num_samples, normalize=False)
                if real_images.min() < 0:
                    real_images = (real_images + 1) / 2
                real_images = real_images.clamp(0, 1)
            
            # Resize real images to match fake images if needed
            fake_h, fake_w = fake_images.shape[-2:]
            real_h, real_w = real_images.shape[-2:]
            
            if (fake_h, fake_w) != (real_h, real_w):
                print(f"      ⚠️ Resizing real images from {real_h}x{real_w} to {fake_h}x{fake_w}")
                real_images = F.interpolate(
                    real_images, 
                    size=(fake_h, fake_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Handle channel mismatch (grayscale vs RGB)
            if fake_images.shape[1] != real_images.shape[1]:
                if fake_images.shape[1] == 1 and real_images.shape[1] == 3:
                    fake_images = fake_images.repeat(1, 3, 1, 1)
                    print(f"      ⚠️ Converted fake images from grayscale to RGB for metrics")
                elif fake_images.shape[1] == 3 and real_images.shape[1] == 1:
                    real_images = real_images.repeat(1, 3, 1, 1)
                    print(f"      ⚠️ Converted real images from grayscale to RGB for metrics")
            
            # ==================== DISTRIBUTION METRICS ====================
            print(f"\n   📊 Computing Distribution Metrics...")
            
            # FID
            print(f"      Computing FID...")
            try:
                fid_result = compute_fid(real_data_dir, str(fake_dir), self.device)
                result.fid = fid_result.value
                print(f"      ✅ FID: {result.fid:.2f}")
            except Exception as e:
                print(f"      ⚠️ FID failed: {e}")
                result.fid = -1.0
            
            # Inception Score
            print(f"      Computing Inception Score...")
            try:
                fake_for_is = fake_images
                if fake_for_is.shape[1] == 1:
                    fake_for_is = fake_for_is.repeat(1, 3, 1, 1)
                is_result = compute_inception_score(fake_for_is, self.device)
                result.is_mean = is_result.value
                result.is_std = is_result.std if is_result.std else 0.0
                print(f"      ✅ IS: {result.is_mean:.2f} ± {result.is_std:.2f}")
            except Exception as e:
                print(f"      ⚠️ IS failed: {e}")
                result.is_mean = -1.0
                result.is_std = 0.0
            
            # KID - only if full metrics and enough samples
            if self.compute_full_metrics:
                print(f"      Computing KID...")
                try:
                    kid_subset = min(1000, num_samples // 2, len(real_images), len(fake_images))
                    if kid_subset >= 10:
                        kid_result = compute_kid(real_images, fake_images, self.device, subset_size=kid_subset)
                        result.kid = kid_result.value
                        print(f"      ✅ KID: {result.kid:.4f}")
                    else:
                        print(f"      ⚠️ KID skipped: not enough samples (need at least 10)")
                except Exception as e:
                    print(f"      ⚠️ KID failed: {e}")
            
            # Precision/Recall
            if self.compute_full_metrics:
                print(f"      Computing Precision/Recall...")
                try:
                    extractor = self._get_feature_extractor()
                    # Ensure RGB for feature extraction
                    fake_for_feat = fake_images if fake_images.shape[1] == 3 else fake_images.repeat(1, 3, 1, 1)
                    real_for_feat = real_images if real_images.shape[1] == 3 else real_images.repeat(1, 3, 1, 1)
                    
                    fake_features = extractor.extract(fake_for_feat)
                    real_features = self._get_real_features(real_for_feat)
                    
                    prec_result, rec_result = compute_precision_recall(
                        real_features, fake_features
                    )
                    result.precision = prec_result.value
                    result.recall = rec_result.value
                    print(f"      ✅ Precision: {result.precision:.4f}")
                    print(f"      ✅ Recall: {result.recall:.4f}")
                except Exception as e:
                    print(f"      ⚠️ Precision/Recall failed: {e}")
            
            # Density & Coverage
            if self.compute_full_metrics:
                print(f"      Computing Density/Coverage...")
                try:
                    if 'fake_features' not in locals():
                        extractor = self._get_feature_extractor()
                        fake_for_feat = fake_images if fake_images.shape[1] == 3 else fake_images.repeat(1, 3, 1, 1)
                        real_for_feat = real_images if real_images.shape[1] == 3 else real_images.repeat(1, 3, 1, 1)
                        fake_features = extractor.extract(fake_for_feat)
                        real_features = self._get_real_features(real_for_feat)
                    
                    dens_result, cov_result = compute_density_coverage(
                        real_features, fake_features
                    )
                    result.density = dens_result.value
                    result.coverage = cov_result.value
                    print(f"      ✅ Density: {result.density:.4f}")
                    print(f"      ✅ Coverage: {result.coverage:.4f}")
                except Exception as e:
                    print(f"      ⚠️ Density/Coverage failed: {e}")
            
            # ==================== PERCEPTUAL METRICS ====================
            if self.compute_full_metrics:
                print(f"\n   🎨 Computing Perceptual Metrics...")
                
                # Use subset for pairwise metrics
                n_pairs = min(500, len(real_images), len(fake_images))
                real_subset = real_images[:n_pairs]
                fake_subset = fake_images[:n_pairs]
                
                # Ensure RGB for perceptual metrics
                if real_subset.shape[1] == 1:
                    real_subset = real_subset.repeat(1, 3, 1, 1)
                if fake_subset.shape[1] == 1:
                    fake_subset = fake_subset.repeat(1, 3, 1, 1)
                
                # LPIPS - ensure [0, 1] range
                print(f"      Computing LPIPS...")
                try:
                    # LPIPS in torchmetrics expects [0, 1] range
                    lpips_result = compute_lpips(real_subset, fake_subset, self.device)
                    result.lpips = lpips_result.value
                    print(f"      ✅ LPIPS: {result.lpips:.4f}")
                except Exception as e:
                    print(f"      ⚠️ LPIPS failed: {e}")
                
                # SSIM
                print(f"      Computing SSIM...")
                try:
                    ssim_result = compute_ssim(real_subset, fake_subset, self.device)
                    result.ssim = ssim_result.value
                    print(f"      ✅ SSIM: {result.ssim:.4f}")
                except Exception as e:
                    print(f"      ⚠️ SSIM failed: {e}")
                
                # PSNR
                print(f"      Computing PSNR...")
                try:
                    psnr_result = compute_psnr(real_subset, fake_subset, self.device)
                    result.psnr = psnr_result.value
                    print(f"      ✅ PSNR: {result.psnr:.2f} dB")
                except Exception as e:
                    print(f"      ⚠️ PSNR failed: {e}")
            
            # ==================== DIVERSITY METRICS ====================
            if self.compute_full_metrics:
                print(f"\n   🌈 Computing Diversity Metrics...")
                
                # Vendi Score
                print(f"      Computing Vendi Score...")
                try:
                    if 'fake_features' not in locals():
                        extractor = self._get_feature_extractor()
                        fake_for_feat = fake_images if fake_images.shape[1] == 3 else fake_images.repeat(1, 3, 1, 1)
                        fake_features = extractor.extract(fake_for_feat)
                    
                    vendi_result = compute_vendi_score(fake_features)
                    result.vendi_score = vendi_result.value
                    print(f"      ✅ Vendi Score: {result.vendi_score:.2f}")
                except Exception as e:
                    print(f"      ⚠️ Vendi Score failed: {e}")
                
                # Intra-LPIPS - ensure [0, 1] range and RGB
                print(f"      Computing Intra-LPIPS...")
                try:
                    fake_for_lpips = fake_images if fake_images.shape[1] == 3 else fake_images.repeat(1, 3, 1, 1)
                    # Ensure [0, 1] range
                    fake_for_lpips = fake_for_lpips.clamp(0, 1)
                    intra_result = compute_intra_lpips(fake_for_lpips, self.device)
                    result.intra_lpips = intra_result.value
                    print(f"      ✅ Intra-LPIPS: {result.intra_lpips:.4f}")
                except Exception as e:
                    print(f"      ⚠️ Intra-LPIPS failed: {e}")
            
            # ==================== EFFICIENCY METRICS ====================
            print(f"\n   ⚡ Computing Efficiency Metrics...")
            
            # FLOPs
            if self.compute_full_metrics:
                print(f"      Computing FLOPs...")
                try:
                    flops_result = compute_flops(
                        model, 
                        (1, *self.image_shape), 
                        self.device
                    )
                    result.flops = flops_result.value
                    print(f"      ✅ FLOPs: {result.flops/1e9:.2f}G")
                except Exception as e:
                    print(f"      ⚠️ FLOPs failed: {e}")
            
            # Memory
            try:
                mem_result = compute_memory_usage(self.device)
                result.memory_gb = mem_result.value
                print(f"      ✅ GPU Memory: {result.memory_gb:.2f} GB")
            except Exception as e:
                print(f"      ⚠️ Memory failed: {e}")
            
            result.success = True
            
        except Exception as e:
            print(f"❌ Strategy failed: {e}")
            import traceback
            traceback.print_exc()
            result.success = False
            result.error = str(e)
        
        # Store result immediately
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
        print(f"\n{'='*70}")
        print(f"🏁 STARTING COMPREHENSIVE BENCHMARK")
        print(f"   Strategies: {len(strategies)}")
        print(f"   Samples per strategy: {num_samples}")
        print(f"   Full metrics: {self.compute_full_metrics}")
        print(f"   Output: {self.output_dir}")
        print(f"{'='*70}")
        
        # Pre-load real images for efficiency
        print(f"\n📂 Pre-loading real images...")
        real_images = load_images_from_dir(real_data_dir, max_images=num_samples, normalize=False)
        
        # Ensure real images are in [0, 1] range
        if real_images.min() < 0:
            real_images = (real_images + 1) / 2
        real_images = real_images.clamp(0, 1)
        
        print(f"   Loaded {len(real_images)} real images")
        
        # Pre-compute real features if using full metrics
        if self.compute_full_metrics:
            print(f"   Pre-computing real features...")
            self._get_real_features(real_images)
        
        for i, strategy in enumerate(strategies):
            print(f"\n[{i+1}/{len(strategies)}]", end="")
            result = self.run_strategy(
                strategy, model, num_samples, batch_size, 
                real_data_dir, real_images.clone()  # Pass a copy to avoid modification
            )
            # Store result with strategy name as key
            self.results[strategy.name] = result
        
        # Save comprehensive report
        report_path = self.output_dir / "report.json"
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_timesteps": self.num_timesteps,
                "image_shape": self.image_shape,
                "full_metrics": self.compute_full_metrics
            },
            "strategies": {name: r.to_dict() for name, r in self.results.items()}
        }
        
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\n✅ Report saved to {report_path}")
        
        # Save CSV summary
        csv_path = self.output_dir / "report.csv"
        
        if self.results:
            # Get fieldnames from first result
            fieldnames = list(next(iter(self.results.values())).to_dict().keys())
            
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results.values():
                    writer.writerow(result.to_dict())
            print(f"📊 CSV saved to {csv_path}")
        
        # Generate plots
        print("\n🎨 Generating plots...")
        try:
            from scripts.plot_results import generate_plots
            generate_plots(self.output_dir)
        except Exception as e:
            print(f"⚠️ Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """Print comprehensive summary table."""
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 100)
        
        if not self.results:
            print("⚠️ No results to display")
            return
        
        # Header
        if self.compute_full_metrics:
            print(f"{'Strategy':<20} {'FID':>8} {'IS':>10} {'Prec':>8} {'Rec':>8} "
                  f"{'Vendi':>8} {'Tput':>12} {'NFE':>6}")
            print("-" * 100)
            
            for name, result in self.results.items():
                fid_str = f"{result.fid:.2f}" if result.fid > 0 else "N/A"
                is_str = f"{result.is_mean:.2f}±{result.is_std:.1f}" if result.is_mean > 0 else "N/A"
                prec_str = f"{result.precision:.3f}" if result.precision > 0 else "N/A"
                rec_str = f"{result.recall:.3f}" if result.recall > 0 else "N/A"
                vendi_str = f"{result.vendi_score:.2f}" if result.vendi_score > 0 else "N/A"
                tp_str = f"{result.throughput:.1f} img/s" if result.throughput > 0 else "N/A"
                nfe_str = str(result.nfe) if result.nfe > 0 else "N/A"
                
                print(f"{name:<20} {fid_str:>8} {is_str:>10} {prec_str:>8} {rec_str:>8} "
                      f"{vendi_str:>8} {tp_str:>12} {nfe_str:>6}")
        else:
            print(f"{'Strategy':<25} {'FID':>10} {'IS':>12} {'Throughput':>15} {'NFE':>8}")
            print("-" * 80)
            
            for name, result in self.results.items():
                fid_str = f"{result.fid:.2f}" if result.fid > 0 else "N/A"
                is_str = f"{result.is_mean:.2f}±{result.is_std:.2f}" if result.is_mean > 0 else "N/A"
                tp_str = f"{result.throughput:.2f} img/s" if result.throughput > 0 else "N/A"
                nfe_str = str(result.nfe) if result.nfe > 0 else "N/A"
                print(f"{name:<25} {fid_str:>10} {is_str:>12} {tp_str:>15} {nfe_str:>8}")
        
        print("=" * 100)
        
        # Find best results
        print("\n🏆 BEST RESULTS:")
        valid_results = [r for r in self.results.values() if r.fid > 0]
        if valid_results:
            best_fid = min(valid_results, key=lambda x: x.fid)
            print(f"   Best FID: {best_fid.name} ({best_fid.fid:.2f})")
        else:
            print("   No valid FID results")
        
        valid_is = [r for r in self.results.values() if r.is_mean > 0]
        if valid_is:
            best_is = max(valid_is, key=lambda x: x.is_mean)
            print(f"   Best IS:  {best_is.name} ({best_is.is_mean:.2f})")
        
        valid_tp = [r for r in self.results.values() if r.throughput > 0]
        if valid_tp:
            best_tp = max(valid_tp, key=lambda x: x.throughput)
            print(f"   Best Throughput: {best_tp.name} ({best_tp.throughput:.2f} img/s)")
        
        print(f"\n📁 Full results: {self.output_dir}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="StepDrop Comprehensive Benchmarking Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  1. Quick test with dummy model:
     python scripts/benchmark_strategies.py --dummy --samples 50

  2. Basic evaluation with trained model:
     python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 1000

  3. Full evaluation with all metrics:
     python scripts/benchmark_strategies.py --checkpoint model.pt --samples 5000 --full-metrics

  4. Specific strategies only:
     python scripts/benchmark_strategies.py --checkpoint model.pt --strategies "DDIM_50,StepDrop_0.3"

METRICS COMPUTED:
  Basic (default):
    - FID (Fréchet Inception Distance)
    - IS (Inception Score)
    - Throughput, NFE
  
  Full (--full-metrics):
    - Distribution: FID, KID, IS, Precision, Recall, Density, Coverage
    - Perceptual: LPIPS, SSIM, PSNR
    - Diversity: Vendi Score, Intra-LPIPS
    - Efficiency: FLOPs, Memory, Throughput, NFE

OUTPUT:
  Results saved to: results/YYYY-MM-DD_HH-MM-SS/
    - report.json (all metrics)
    - report.csv (for easy analysis)
    - <strategy>/samples/*.png (generated images)
"""
    )
    
    # Model
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy model (for testing)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    
    # Evaluation
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples per strategy")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation")
    parser.add_argument("--full-metrics", action="store_true",
                        help="Compute all metrics (slower but comprehensive)")
    
    # Strategies
    parser.add_argument("--strategies", type=str, default="all",
                        help="Comma-separated strategy names or 'all'")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--real_data_dir", type=str, default="data/fid_real_cifar10",
                        help="Directory with real images for FID")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Load model (now returns config too)
    model, model_config = load_model(args.checkpoint, device, dummy=args.dummy)
    
    # Determine image shape from model config
    image_shape = (model_config['channels'], model_config['img_size'], model_config['img_size'])
    
    # Prepare real data based on channels
    if model_config['channels'] == 1:
        # MNIST - need to prepare MNIST real data
        print("📊 Detected grayscale model (likely MNIST)")
        real_data_dir = args.real_data_dir.replace("cifar10", "mnist") if "cifar10" in args.real_data_dir else args.real_data_dir
        # For now, skip real data for MNIST or prepare it differently
        num_real = 100 if args.dummy else min(10000, args.samples * 2)
        # You may need to implement save_mnist_real_subset similar to save_cifar10_real_subset
        try:
            real_data_dir = prepare_real_data(
                out_dir=real_data_dir,
                num_images=num_real,
                dataset="mnist" if model_config['channels'] == 1 else "cifar10"
            )
        except Exception as e:
            print(f"⚠️ Could not prepare real data: {e}")
            print("   Using CIFAR-10 fallback (metrics may be invalid for MNIST model)")
            num_real = 100 if args.dummy else min(10000, args.samples * 2)
            real_data_dir = prepare_real_data(
                out_dir=args.real_data_dir,
                num_images=num_real
            )
    else:
        # CIFAR-10 or other RGB
        num_real = 100 if args.dummy else min(10000, args.samples * 2)
        real_data_dir = prepare_real_data(
            out_dir=args.real_data_dir,
            num_images=num_real
        )
    
    # Select strategies
    if args.strategies == "all":
        strategies = DEFAULT_STRATEGIES
    else:
        names = [s.strip() for s in args.strategies.split(",")]
        strategies = [s for s in DEFAULT_STRATEGIES if s.name in names]
        if not strategies:
            print(f"⚠️ No matching strategies found. Available: {[s.name for s in DEFAULT_STRATEGIES]}")
            return
    
    # For dummy/quick runs, limit strategies
    if args.dummy:
        strategies = strategies[:3]
        print(f"🎭 Dummy mode: Using {len(strategies)} strategies, {args.samples} samples")
    
    # Run benchmark with correct image shape
    runner = ComprehensiveBenchmarkRunner(
        output_dir=args.output_dir,
        device=device,
        compute_full_metrics=args.full_metrics,
        image_shape=image_shape  # Pass the correct image shape
    )
    
    runner.run_all(
        strategies=strategies,
        model=model,
        num_samples=args.samples,
        batch_size=args.batch_size,
        real_data_dir=real_data_dir
    )
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
