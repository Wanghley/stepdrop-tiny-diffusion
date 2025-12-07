"""
Comprehensive Metrics Utility Module for Diffusion Model Evaluation

This module provides a unified interface for calculating standard generative model metrics
used in state-of-the-art diffusion model papers (DDPM, DDIM, Stable Diffusion, etc.).

Supported Metrics:
==================

DISTRIBUTION METRICS (Real vs Generated):
1.  FID (Fréchet Inception Distance) - Lower is better
    - Gold standard for generative models
    - Compares statistics of Inception-v3 features
    - CIFAR-10 SOTA: ~2-3, Good: <10, Acceptable: <50

2.  KID (Kernel Inception Distance) - Lower is better  
    - Unbiased alternative to FID
    - Better for small sample sizes
    - Uses MMD with polynomial kernel

3.  IS (Inception Score) - Higher is better
    - Measures quality and diversity
    - CIFAR-10 real data: ~11.0

4.  Precision & Recall - Higher is better
    - Precision: Quality (do generated images look real?)
    - Recall: Diversity (does generator cover the real distribution?)

5.  Density & Coverage - Higher is better
    - Improved version of Precision/Recall
    - More robust to outliers

PERCEPTUAL METRICS (Image Quality):
6.  LPIPS (Learned Perceptual Image Patch Similarity) - Lower is better
    - Perceptual similarity using deep features
    - Correlates well with human perception

7.  SSIM (Structural Similarity Index) - Higher is better
    - Measures structural similarity
    - Range: [-1, 1], typically [0, 1]

8.  PSNR (Peak Signal-to-Noise Ratio) - Higher is better
    - Measures reconstruction quality
    - Good: >30dB, Excellent: >40dB

9.  MS-SSIM (Multi-Scale SSIM) - Higher is better
    - Multi-scale version of SSIM
    - More robust across scales

DIVERSITY METRICS:
10. VENDI Score - Higher is better
    - Measures diversity using eigenvalues
    - Based on matrix-based entropy

11. Intra-FID / Intra-LPIPS
    - Measures diversity within generated samples

EFFICIENCY METRICS:
12. FLOPs (Floating Point Operations) - Lower is better
13. Throughput (Images/Second) - Higher is better
14. Memory Usage (GB) - Lower is better
15. NFE (Number of Function Evaluations) - Lower is better

References:
- FID: https://arxiv.org/abs/1706.08500
- KID: https://arxiv.org/abs/1801.01401
- IS: https://arxiv.org/abs/1606.03498
- LPIPS: https://arxiv.org/abs/1801.03924
- Precision/Recall: https://arxiv.org/abs/1904.06991
- Density/Coverage: https://arxiv.org/abs/2002.09797
"""

from typing import Callable, Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import math
import json
import time
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

# Optional imports with graceful fallbacks
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    from pytorch_fid.inception import InceptionV3
    HAS_PYTORCH_FID = True
except ImportError:
    HAS_PYTORCH_FID = False
    warnings.warn("pytorch-fid not installed. FID computation will be limited.")

try:
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.functional.image import (
        structural_similarity_index_measure as ssim_fn,
        peak_signal_noise_ratio as psnr_fn,
        multiscale_structural_similarity_index_measure as ms_ssim_fn,
    )
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    warnings.warn("torchmetrics not fully installed. Some metrics will be unavailable.")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from cleanfid import fid as cleanfid
    HAS_CLEANFID = True
except ImportError:
    HAS_CLEANFID = False


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class MetricResult:
    """Container for a single metric result."""
    name: str
    value: float
    std: Optional[float] = None
    unit: str = ""
    higher_is_better: bool = False
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "std": self.std,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "description": self.description
        }
    
    def __str__(self) -> str:
        direction = "↑" if self.higher_is_better else "↓"
        if self.std is not None:
            return f"{self.name}: {self.value:.4f} ± {self.std:.4f} {self.unit} ({direction})"
        return f"{self.name}: {self.value:.4f} {self.unit} ({direction})"


@dataclass 
class EvaluationReport:
    """Complete evaluation report with all metrics."""
    
    # Distribution metrics
    fid: Optional[MetricResult] = None
    kid: Optional[MetricResult] = None
    inception_score: Optional[MetricResult] = None
    precision: Optional[MetricResult] = None
    recall: Optional[MetricResult] = None
    density: Optional[MetricResult] = None
    coverage: Optional[MetricResult] = None
    
    # Perceptual metrics
    lpips: Optional[MetricResult] = None
    ssim: Optional[MetricResult] = None
    psnr: Optional[MetricResult] = None
    ms_ssim: Optional[MetricResult] = None
    
    # Diversity metrics
    vendi_score: Optional[MetricResult] = None
    intra_lpips: Optional[MetricResult] = None
    
    # Efficiency metrics
    flops: Optional[MetricResult] = None
    throughput: Optional[MetricResult] = None
    memory_gb: Optional[MetricResult] = None
    nfe: Optional[MetricResult] = None
    
    # Metadata
    num_samples: int = 0
    timestamp: str = ""
    device: str = ""
    model_name: str = ""
    strategy_name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        result = {
            "metadata": {
                "num_samples": self.num_samples,
                "timestamp": self.timestamp,
                "device": self.device,
                "model_name": self.model_name,
                "strategy_name": self.strategy_name,
                "config": self.config
            },
            "distribution_metrics": {},
            "perceptual_metrics": {},
            "diversity_metrics": {},
            "efficiency_metrics": {}
        }
        
        # Distribution metrics
        for name in ["fid", "kid", "inception_score", "precision", "recall", "density", "coverage"]:
            metric = getattr(self, name)
            if metric is not None:
                result["distribution_metrics"][name] = metric.to_dict()
        
        # Perceptual metrics
        for name in ["lpips", "ssim", "psnr", "ms_ssim"]:
            metric = getattr(self, name)
            if metric is not None:
                result["perceptual_metrics"][name] = metric.to_dict()
        
        # Diversity metrics
        for name in ["vendi_score", "intra_lpips"]:
            metric = getattr(self, name)
            if metric is not None:
                result["diversity_metrics"][name] = metric.to_dict()
        
        # Efficiency metrics
        for name in ["flops", "throughput", "memory_gb", "nfe"]:
            metric = getattr(self, name)
            if metric is not None:
                result["efficiency_metrics"][name] = metric.to_dict()
        
        return result
    
    def save_json(self, path: str):
        """Save report to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self):
        """Print formatted summary of results."""
        print("\n" + "=" * 70)
        print(f"📊 EVALUATION REPORT: {self.strategy_name}")
        print("=" * 70)
        print(f"Samples: {self.num_samples} | Device: {self.device} | Model: {self.model_name}")
        print("-" * 70)
        
        print("\n📈 DISTRIBUTION METRICS (Real vs Generated):")
        for name in ["fid", "kid", "inception_score", "precision", "recall", "density", "coverage"]:
            metric = getattr(self, name)
            if metric is not None:
                print(f"  {metric}")
        
        print("\n🎨 PERCEPTUAL METRICS:")
        for name in ["lpips", "ssim", "psnr", "ms_ssim"]:
            metric = getattr(self, name)
            if metric is not None:
                print(f"  {metric}")
        
        print("\n🌈 DIVERSITY METRICS:")
        for name in ["vendi_score", "intra_lpips"]:
            metric = getattr(self, name)
            if metric is not None:
                print(f"  {metric}")
        
        print("\n⚡ EFFICIENCY METRICS:")
        for name in ["flops", "throughput", "memory_gb", "nfe"]:
            metric = getattr(self, name)
            if metric is not None:
                print(f"  {metric}")
        
        print("=" * 70 + "\n")


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_images_from_dir(
    image_dir: str,
    max_images: Optional[int] = None,
    image_size: Optional[int] = None,
    normalize: bool = True
) -> torch.Tensor:
    """Load images from directory into tensor."""
    path = Path(image_dir)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = [f for f in path.iterdir() if f.suffix.lower() in extensions]
    
    if max_images:
        files = files[:max_images]
    
    transform_list = []
    if image_size:
        transform_list.append(transforms.Resize((image_size, image_size)))
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize([0.5]*3, [0.5]*3))
    
    transform = transforms.Compose(transform_list)
    
    images = []
    for f in tqdm(files, desc="Loading images", leave=False):
        img = datasets.folder.default_loader(str(f))
        img = transform(img)
        images.append(img)
    
    return torch.stack(images)


def tensor_to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] or [0, 1] to uint8 [0, 255]."""
    if images.min() < 0:  # Assume [-1, 1]
        images = (images + 1) / 2
    images = (images * 255).clamp(0, 255).byte()
    return images


# =============================================================================
# Dataset Preparation
# =============================================================================

def save_cifar10_real_subset(
    out_dir: str = "data/fid_real_cifar10",
    split: str = "train",
    num_images: int = 10000,
    data_dir: str = "data",
    batch_size: int = 256,
    num_workers: int = 4,
) -> Path:
    """Save subset of CIFAR-10 images as PNGs for FID reference."""
    out_path = ensure_dir(out_dir)
    
    # Check if already generated enough images
    num_existing = len(list(out_path.glob("*.png")))
    if num_existing >= num_images:
        print(f"✓ Real data cache exists: {out_path} ({num_existing} images)")
        return out_path
    
    print(f"Preparing {num_images} CIFAR-10 real images...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=(split == "train"),
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    saved_count = 0
    pbar = tqdm(total=num_images, desc="Saving real images")
    
    for images, _ in loader:
        for img in images:
            if saved_count >= num_images:
                break
            save_image(img, out_path / f"{saved_count:06d}.png", normalize=True, value_range=(-1, 1))
            saved_count += 1
            pbar.update(1)
        if saved_count >= num_images:
            break
    
    pbar.close()
    print(f"✓ Saved {saved_count} real images to {out_path}")
    return out_path


@torch.no_grad()
def save_fake_images(
    generator: Callable[[int], torch.Tensor],
    out_dir: str = "data/fid_fake_samples",
    num_images: int = 10000,
    batch_size: int = 64,
    device: str = "cuda",
    clear_dir: bool = True
) -> Path:
    """Generate and save fake images for evaluation."""
    out_path = ensure_dir(out_dir)
    
    if clear_dir:
        for f in out_path.glob("*.png"):
            f.unlink()
    
    saved_count = 0
    pbar = tqdm(total=num_images, desc="Generating fake images")
    
    while saved_count < num_images:
        current_batch = min(batch_size, num_images - saved_count)
        images = generator(current_batch)
        
        if isinstance(images, tuple):
            images = images[0]  # Handle (images, stats) return
        
        images = images.to(device)
        
        for img in images:
            save_image(img, out_path / f"{saved_count:06d}.png", normalize=True, value_range=(-1, 1))
            saved_count += 1
            pbar.update(1)
    
    pbar.close()
    return out_path


# =============================================================================
# Distribution Metrics
# =============================================================================

@torch.no_grad()
def compute_fid(
    real_dir: str,
    fake_dir: str,
    device: str = "cuda",
    batch_size: int = 64
) -> MetricResult:
    """
    Compute Fréchet Inception Distance (FID).
    
    FID measures the distance between the distributions of real and generated images
    in the feature space of Inception-v3.
    
    Formula: FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^(1/2))
    """
    if not HAS_PYTORCH_FID:
        raise ImportError("pytorch-fid required. Install with: pip install pytorch-fid")
    
    fid_value = calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=2048
    )
    
    return MetricResult(
        name="FID",
        value=fid_value,
        higher_is_better=False,
        description="Fréchet Inception Distance - measures distribution similarity"
    )


@torch.no_grad()
def compute_clean_fid(
    real_dir: str,
    fake_dir: str,
    mode: str = "clean",  # "clean", "legacy_pytorch", "legacy_tensorflow"
    device: str = "cuda"
) -> MetricResult:
    """
    Compute Clean-FID (more accurate FID implementation).
    
    Clean-FID addresses inconsistencies in FID computation across implementations.
    Reference: https://arxiv.org/abs/2104.11222
    """
    if not HAS_CLEANFID:
        raise ImportError("clean-fid required. Install with: pip install clean-fid")
    
    fid_value = cleanfid.compute_fid(
        real_dir,
        fake_dir,
        mode=mode,
        device=device
    )
    
    return MetricResult(
        name="Clean-FID",
        value=fid_value,
        higher_is_better=False,
        description=f"Clean-FID ({mode}) - more accurate FID implementation"
    )


@torch.no_grad()
def compute_kid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = "cuda",
    subset_size: int = 1000
) -> MetricResult:
    """
    Compute Kernel Inception Distance (KID).
    
    KID is an unbiased estimator that works better with small sample sizes.
    Uses MMD (Maximum Mean Discrepancy) with polynomial kernel.
    """
    if not HAS_TORCHMETRICS:
        raise ImportError("torchmetrics required. Install with: pip install torchmetrics")
    
    kid = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(device)
    
    # Ensure uint8 format
    real_uint8 = tensor_to_uint8(real_images).to(device)
    fake_uint8 = tensor_to_uint8(fake_images).to(device)
    
    kid.update(real_uint8, real=True)
    kid.update(fake_uint8, real=False)
    
    mean, std = kid.compute()
    
    return MetricResult(
        name="KID",
        value=mean.item(),
        std=std.item(),
        higher_is_better=False,
        description="Kernel Inception Distance - unbiased alternative to FID"
    )


@torch.no_grad()
def compute_inception_score(
    images: torch.Tensor,
    device: str = "cuda",
    splits: int = 10
) -> MetricResult:
    """
    Compute Inception Score (IS).
    
    IS measures quality (low entropy p(y|x)) and diversity (high entropy p(y)).
    Formula: IS = exp(E[KL(p(y|x) || p(y))])
    
    CIFAR-10 reference: Real data IS ≈ 11.0
    """
    if not HAS_TORCHMETRICS:
        raise ImportError("torchmetrics required")
    
    inception = InceptionScore(normalize=True).to(device)
    
    images_uint8 = tensor_to_uint8(images).to(device)
    inception.update(images_uint8)
    
    mean, std = inception.compute()
    
    return MetricResult(
        name="IS",
        value=mean.item(),
        std=std.item(),
        higher_is_better=True,
        description="Inception Score - measures quality and diversity"
    )


@torch.no_grad()
def compute_precision_recall(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    k: int = 3
) -> Tuple[MetricResult, MetricResult]:
    """
    Compute Improved Precision and Recall.
    
    Precision: What fraction of generated images are realistic?
    Recall: What fraction of real images can the generator produce?
    
    Reference: https://arxiv.org/abs/1904.06991
    """
    # Compute pairwise distances
    def compute_nearest_neighbors(features, k):
        distances = torch.cdist(features, features)
        distances.fill_diagonal_(float('inf'))
        kth_distances, _ = torch.topk(distances, k, largest=False, dim=1)
        return kth_distances[:, -1]  # k-th nearest neighbor distance
    
    real_radii = compute_nearest_neighbors(real_features, k)
    
    # Precision: fraction of fake samples within real manifold
    fake_to_real = torch.cdist(fake_features, real_features)
    min_distances_to_real = fake_to_real.min(dim=1).values
    precision = (min_distances_to_real <= real_radii.unsqueeze(0).min(dim=1).values).float().mean()
    
    # Recall: fraction of real samples covered by fake manifold
    fake_radii = compute_nearest_neighbors(fake_features, k)
    real_to_fake = torch.cdist(real_features, fake_features)
    min_distances_to_fake = real_to_fake.min(dim=1).values
    recall = (min_distances_to_fake <= fake_radii.unsqueeze(0).min(dim=1).values).float().mean()
    
    precision_result = MetricResult(
        name="Precision",
        value=precision.item(),
        higher_is_better=True,
        description="Fraction of generated images that look realistic"
    )
    
    recall_result = MetricResult(
        name="Recall",
        value=recall.item(),
        higher_is_better=True,
        description="Fraction of real distribution covered by generator"
    )
    
    return precision_result, recall_result


@torch.no_grad()
def compute_density_coverage(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    k: int = 5
) -> Tuple[MetricResult, MetricResult]:
    """
    Compute Density and Coverage metrics.
    
    More robust version of Precision/Recall using manifold estimation.
    Reference: https://arxiv.org/abs/2002.09797
    """
    def get_knn_distances(features, k):
        distances = torch.cdist(features, features)
        distances.fill_diagonal_(float('inf'))
        knn_distances, _ = torch.topk(distances, k, largest=False, dim=1)
        return knn_distances[:, -1]  # k-th nearest neighbor
    
    real_knn = get_knn_distances(real_features, k)
    
    # Density: average number of real samples in the neighborhood of each fake sample
    fake_to_real = torch.cdist(fake_features, real_features)
    density_counts = (fake_to_real <= real_knn.unsqueeze(0)).float().sum(dim=1)
    density = density_counts.mean() / k
    
    # Coverage: fraction of real samples with at least one fake neighbor
    real_to_fake = torch.cdist(real_features, fake_features)
    min_fake_dist = real_to_fake.min(dim=1).values
    coverage = (min_fake_dist <= real_knn).float().mean()
    
    density_result = MetricResult(
        name="Density",
        value=density.item(),
        higher_is_better=True,
        description="Average density of fake samples near real manifold"
    )
    
    coverage_result = MetricResult(
        name="Coverage",
        value=coverage.item(),
        higher_is_better=True,
        description="Fraction of real distribution covered"
    )
    
    return density_result, coverage_result


# =============================================================================
# Perceptual Metrics
# =============================================================================

@torch.no_grad()
def compute_lpips(
    images1: torch.Tensor,
    images2: torch.Tensor,
    device: str = "cuda",
    net: str = "alex"  # "alex", "vgg", "squeeze"
) -> MetricResult:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).
    
    Uses deep features to measure perceptual similarity.
    Reference: https://arxiv.org/abs/1801.03924
    """
    if HAS_LPIPS:
        loss_fn = lpips.LPIPS(net=net).to(device)
        
        # Ensure [-1, 1] range
        if images1.min() >= 0:
            images1 = images1 * 2 - 1
        if images2.min() >= 0:
            images2 = images2 * 2 - 1
        
        images1 = images1.to(device)
        images2 = images2.to(device)
        
        lpips_values = loss_fn(images1, images2)
        mean_lpips = lpips_values.mean().item()
        std_lpips = lpips_values.std().item()
        
    elif HAS_TORCHMETRICS:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).to(device)
        
        images1 = images1.to(device)
        images2 = images2.to(device)
        
        mean_lpips = lpips_metric(images1, images2).item()
        std_lpips = 0.0
    else:
        raise ImportError("lpips or torchmetrics required")
    
    return MetricResult(
        name="LPIPS",
        value=mean_lpips,
        std=std_lpips,
        higher_is_better=False,
        description=f"Learned Perceptual Similarity ({net})"
    )


@torch.no_grad()
def compute_ssim(
    images1: torch.Tensor,
    images2: torch.Tensor,
    device: str = "cuda"
) -> MetricResult:
    """
    Compute SSIM (Structural Similarity Index).
    
    Measures structural similarity between images.
    Range: [-1, 1], typically [0, 1] for natural images.
    """
    if not HAS_TORCHMETRICS:
        raise ImportError("torchmetrics required")
    
    images1 = images1.to(device)
    images2 = images2.to(device)
    
    # Ensure [0, 1] range
    if images1.min() < 0:
        images1 = (images1 + 1) / 2
    if images2.min() < 0:
        images2 = (images2 + 1) / 2
    
    ssim_value = ssim_fn(images1, images2, data_range=1.0)
    
    return MetricResult(
        name="SSIM",
        value=ssim_value.item(),
        higher_is_better=True,
        description="Structural Similarity Index"
    )


@torch.no_grad()
def compute_psnr(
    images1: torch.Tensor,
    images2: torch.Tensor,
    device: str = "cuda"
) -> MetricResult:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio).
    
    Good: >30dB, Excellent: >40dB
    """
    if not HAS_TORCHMETRICS:
        raise ImportError("torchmetrics required")
    
    images1 = images1.to(device)
    images2 = images2.to(device)
    
    # Ensure [0, 1] range
    if images1.min() < 0:
        images1 = (images1 + 1) / 2
    if images2.min() < 0:
        images2 = (images2 + 1) / 2
    
    psnr_value = psnr_fn(images1, images2, data_range=1.0)
    
    return MetricResult(
        name="PSNR",
        value=psnr_value.item(),
        unit="dB",
        higher_is_better=True,
        description="Peak Signal-to-Noise Ratio"
    )


@torch.no_grad()
def compute_ms_ssim(
    images1: torch.Tensor,
    images2: torch.Tensor,
    device: str = "cuda"
) -> MetricResult:
    """
    Compute MS-SSIM (Multi-Scale SSIM).
    
    More robust than SSIM across different scales.
    """
    if not HAS_TORCHMETRICS:
        raise ImportError("torchmetrics required")
    
    images1 = images1.to(device)
    images2 = images2.to(device)
    
    if images1.min() < 0:
        images1 = (images1 + 1) / 2
    if images2.min() < 0:
        images2 = (images2 + 1) / 2
    
    # MS-SSIM requires minimum size of 160x160 for 5 scales
    # For smaller images, we'll use fewer scales
    min_size = min(images1.shape[-2:])
    if min_size < 160:
        # Use regular SSIM for small images
        return compute_ssim(images1, images2, device)
    
    ms_ssim_value = ms_ssim_fn(images1, images2, data_range=1.0)
    
    return MetricResult(
        name="MS-SSIM",
        value=ms_ssim_value.item(),
        higher_is_better=True,
        description="Multi-Scale Structural Similarity"
    )


# =============================================================================
# Diversity Metrics
# =============================================================================

@torch.no_grad()
def compute_vendi_score(
    features: torch.Tensor,
    q: float = 1.0
) -> MetricResult:
    """
    Compute VENDI Score (diversity metric based on eigenvalues).
    
    Measures diversity using matrix-based entropy of the similarity matrix.
    Reference: https://arxiv.org/abs/2210.02410
    """
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity = torch.mm(features, features.t())
    
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(similarity)
    eigenvalues = eigenvalues.real.clamp(min=1e-10)
    eigenvalues = eigenvalues / eigenvalues.sum()  # Normalize
    
    # Compute VENDI score (exponential of entropy)
    if q == 1:
        # Shannon entropy
        entropy = -(eigenvalues * torch.log(eigenvalues)).sum()
    else:
        # Rényi entropy
        entropy = (1 / (1 - q)) * torch.log((eigenvalues ** q).sum())
    
    vendi = torch.exp(entropy).item()
    
    return MetricResult(
        name="VENDI",
        value=vendi,
        higher_is_better=True,
        description=f"VENDI Score (q={q}) - eigenvalue-based diversity"
    )


@torch.no_grad()
def compute_intra_lpips(
    images: torch.Tensor,
    device: str = "cuda",
    num_pairs: int = 1000
) -> MetricResult:
    """
    Compute Intra-LPIPS (diversity within generated samples).
    
    Higher values indicate more diverse generations.
    """
    if not (HAS_LPIPS or HAS_TORCHMETRICS):
        raise ImportError("lpips or torchmetrics required")
    
    n = len(images)
    num_pairs = min(num_pairs, n * (n - 1) // 2)
    
    if HAS_LPIPS:
        loss_fn = lpips.LPIPS(net='alex').to(device)
    else:
        loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    
    images = images.to(device)
    if images.min() >= 0:
        images = images * 2 - 1
    
    # Random pairs
    indices = torch.randperm(n * (n - 1) // 2)[:num_pairs]
    
    lpips_values = []
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pair_idx in indices:
                val = loss_fn(images[i:i+1], images[j:j+1])
                lpips_values.append(val.item())
            pair_idx += 1
            if len(lpips_values) >= num_pairs:
                break
        if len(lpips_values) >= num_pairs:
            break
    
    mean_lpips = np.mean(lpips_values)
    std_lpips = np.std(lpips_values)
    
    return MetricResult(
        name="Intra-LPIPS",
        value=mean_lpips,
        std=std_lpips,
        higher_is_better=True,
        description="Intra-sample LPIPS diversity"
    )


# =============================================================================
# Efficiency Metrics
# =============================================================================

def compute_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda"
) -> MetricResult:
    """Compute FLOPs for a single forward pass."""
    if not HAS_THOP:
        raise ImportError("thop required. Install with: pip install thop")
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_shape).to(device)
    dummy_timestep = torch.tensor([0]).to(device)
    
    flops, params = profile(model, inputs=(dummy_input, dummy_timestep), verbose=False)
    
    return MetricResult(
        name="FLOPs",
        value=flops,
        unit="",
        higher_is_better=False,
        description=f"Floating point operations per forward pass ({params/1e6:.2f}M params)"
    )


def compute_throughput(
    generator: Callable,
    num_images: int = 100,
    batch_size: int = 16,
    warmup_batches: int = 2,
    device: str = "cuda"
) -> MetricResult:
    """Measure generation throughput in images/second."""
    
    # Warmup
    for _ in range(warmup_batches):
        _ = generator(batch_size)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    generated = 0
    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)
        _ = generator(current_batch)
        generated += current_batch
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    throughput = num_images / elapsed
    
    return MetricResult(
        name="Throughput",
        value=throughput,
        unit="img/s",
        higher_is_better=True,
        description="Generation speed"
    )


def compute_memory_usage(device: str = "cuda") -> MetricResult:
    """Measure peak GPU memory usage."""
    if device != "cuda" or not torch.cuda.is_available():
        return MetricResult(
            name="Memory",
            value=0,
            unit="GB",
            higher_is_better=False,
            description="GPU memory usage (N/A for CPU)"
        )
    
    torch.cuda.reset_peak_memory_stats()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    return MetricResult(
        name="Memory",
        value=peak_memory,
        unit="GB",
        higher_is_better=False,
        description="Peak GPU memory usage"
    )


# =============================================================================
# Feature Extraction (for Precision/Recall, Density/Coverage)
# =============================================================================

class InceptionFeatureExtractor:
    """Extract features using Inception-v3 for metric computation."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        if HAS_PYTORCH_FID:
            # Use pytorch-fid's Inception for consistency with FID
            self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        else:
            # Fallback to torchvision
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
            self.model.fc = nn.Identity()
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract(self, images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """Extract features from images."""
        features_list = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            
            # Ensure [0, 1] range
            if batch.min() < 0:
                batch = (batch + 1) / 2
            
            # Resize and normalize
            batch = self.transform(batch)
            
            # Extract features
            if HAS_PYTORCH_FID:
                features = self.model(batch)[0]
            else:
                features = self.model(batch)
            
            features = features.squeeze()
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            features_list.append(features.cpu())
        
        return torch.cat(features_list, dim=0)


# =============================================================================
# Comprehensive Evaluator
# =============================================================================

class DiffusionEvaluator:
    """
    Comprehensive evaluator for diffusion models.
    
    Computes all standard metrics used in diffusion model papers.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        real_data_dir: Optional[str] = None,
        num_real_samples: int = 10000
    ):
        self.device = device
        self.real_data_dir = real_data_dir
        self.num_real_samples = num_real_samples
        self.feature_extractor = None
        self._real_features = None
    
    def _get_feature_extractor(self) -> InceptionFeatureExtractor:
        if self.feature_extractor is None:
            self.feature_extractor = InceptionFeatureExtractor(self.device)
        return self.feature_extractor
    
    def _get_real_features(self, real_images: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._real_features is not None:
            return self._real_features
        
        if real_images is not None:
            extractor = self._get_feature_extractor()
            self._real_features = extractor.extract(real_images)
        elif self.real_data_dir:
            real_images = load_images_from_dir(
                self.real_data_dir,
                max_images=self.num_real_samples
            )
            extractor = self._get_feature_extractor()
            self._real_features = extractor.extract(real_images)
        
        return self._real_features
    
    @torch.no_grad()
    def evaluate(
        self,
        generator: Optional[Callable[[int], torch.Tensor]] = None,
        fake_images: Optional[torch.Tensor] = None,
        fake_dir: Optional[str] = None,
        real_images: Optional[torch.Tensor] = None,
        real_dir: Optional[str] = None,
        num_samples: int = 10000,
        batch_size: int = 64,
        compute_fid: bool = True,
        compute_kid: bool = True,
        compute_is: bool = True,
        compute_precision_recall: bool = True,
        compute_density_coverage: bool = True,
        compute_lpips: bool = False,
        compute_ssim: bool = False,
        compute_diversity: bool = True,
        compute_efficiency: bool = True,
        model: Optional[nn.Module] = None,
        model_input_shape: Optional[Tuple[int, ...]] = None,
        strategy_name: str = "Unknown",
        model_name: str = "Unknown",
        nfe: Optional[int] = None
    ) -> EvaluationReport:
        """
        Run comprehensive evaluation.
        
        Args:
            generator: Function that generates images given batch size
            fake_images: Pre-generated fake images tensor
            fake_dir: Directory containing fake images
            real_images: Real images tensor
            real_dir: Directory containing real images
            num_samples: Number of samples for evaluation
            batch_size: Batch size for generation
            compute_*: Flags to enable/disable specific metrics
            model: Model for FLOPs computation
            model_input_shape: Input shape for FLOPs computation
            strategy_name: Name of sampling strategy
            model_name: Name of model
            nfe: Number of function evaluations (denoising steps)
        
        Returns:
            EvaluationReport with all computed metrics
        """
        from datetime import datetime
        
        report = EvaluationReport(
            num_samples=num_samples,
            timestamp=datetime.now().isoformat(),
            device=self.device,
            model_name=model_name,
            strategy_name=strategy_name
        )
        
        # Prepare fake images
        if fake_images is None:
            if fake_dir:
                fake_images = load_images_from_dir(fake_dir, max_images=num_samples)
            elif generator:
                print("Generating fake images...")
                fake_dir = save_fake_images(
                    generator,
                    num_images=num_samples,
                    batch_size=batch_size,
                    device=self.device
                )
                fake_images = load_images_from_dir(str(fake_dir), max_images=num_samples)
        
        # Prepare real data
        real_dir = real_dir or self.real_data_dir
        if real_images is None and real_dir:
            real_images = load_images_from_dir(real_dir, max_images=num_samples)
        
        # Extract features for distribution metrics
        if compute_kid or compute_precision_recall or compute_density_coverage or compute_diversity:
            print("Extracting features...")
            extractor = self._get_feature_extractor()
            fake_features = extractor.extract(fake_images)
            if real_images is not None:
                real_features = self._get_real_features(real_images)
        
        # ===== Distribution Metrics =====
        if compute_fid and real_dir and fake_dir:
            print("Computing FID...")
            try:
                report.fid = compute_fid(str(real_dir), str(fake_dir), self.device)
            except Exception as e:
                print(f"  ⚠️ FID failed: {e}")
        
        if compute_kid and real_images is not None:
            print("Computing KID...")
            try:
                report.kid = compute_kid(real_images, fake_images, self.device)
            except Exception as e:
                print(f"  ⚠️ KID failed: {e}")
        
        if compute_is:
            print("Computing Inception Score...")
            try:
                report.inception_score = compute_inception_score(fake_images, self.device)
            except Exception as e:
                print(f"  ⚠️ IS failed: {e}")
        
        if compute_precision_recall and real_images is not None:
            print("Computing Precision/Recall...")
            try:
                report.precision, report.recall = compute_precision_recall(
                    real_features, fake_features
                )
            except Exception as e:
                print(f"  ⚠️ Precision/Recall failed: {e}")
        
        if compute_density_coverage and real_images is not None:
            print("Computing Density/Coverage...")
            try:
                report.density, report.coverage = compute_density_coverage(
                    real_features, fake_features
                )
            except Exception as e:
                print(f"  ⚠️ Density/Coverage failed: {e}")
        
        # ===== Perceptual Metrics =====
        if compute_lpips and real_images is not None:
            print("Computing LPIPS...")
            try:
                # Compare random pairs of real vs fake
                n = min(len(real_images), len(fake_images), 1000)
                report.lpips = compute_lpips(
                    real_images[:n], fake_images[:n], self.device
                )
            except Exception as e:
                print(f"  ⚠️ LPIPS failed: {e}")
        
        if compute_ssim and real_images is not None:
            print("Computing SSIM...")
            try:
                n = min(len(real_images), len(fake_images), 1000)
                report.ssim = compute_ssim(
                    real_images[:n], fake_images[:n], self.device
                )
            except Exception as e:
                print(f"  ⚠️ SSIM failed: {e}")
        
        # ===== Diversity Metrics =====
        if compute_diversity:
            print("Computing diversity metrics...")
            try:
                report.vendi_score = compute_vendi_score(fake_features)
            except Exception as e:
                print(f"  ⚠️ VENDI failed: {e}")
            
            try:
                report.intra_lpips = compute_intra_lpips(fake_images, self.device)
            except Exception as e:
                print(f"  ⚠️ Intra-LPIPS failed: {e}")
        
        # ===== Efficiency Metrics =====
        if compute_efficiency:
            print("Computing efficiency metrics...")
            
            if model and model_input_shape:
                try:
                    report.flops = compute_flops(model, model_input_shape, self.device)
                except Exception as e:
                    print(f"  ⚠️ FLOPs failed: {e}")
            
            if generator:
                try:
                    report.throughput = compute_throughput(
                        generator, num_images=100, batch_size=batch_size, device=self.device
                    )
                except Exception as e:
                    print(f"  ⚠️ Throughput failed: {e}")
            
            try:
                report.memory_gb = compute_memory_usage(self.device)
            except Exception as e:
                print(f"  ⚠️ Memory failed: {e}")
            
            if nfe is not None:
                report.nfe = MetricResult(
                    name="NFE",
                    value=nfe,
                    higher_is_better=False,
                    description="Number of function evaluations (denoising steps)"
                )
        
        return report


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_evaluate(
    fake_dir: str,
    real_dir: str,
    device: str = "cuda"
) -> Dict[str, float]:
    """Quick evaluation with just FID and IS."""
    results = {}
    
    try:
        fid_result = compute_fid(real_dir, fake_dir, device)
        results["fid"] = fid_result.value
    except Exception as e:
        print(f"FID failed: {e}")
        results["fid"] = -1
    
    try:
        fake_images = load_images_from_dir(fake_dir)
        is_result = compute_inception_score(fake_images, device)
        results["is_mean"] = is_result.value
        results["is_std"] = is_result.std
    except Exception as e:
        print(f"IS failed: {e}")
        results["is_mean"] = -1
        results["is_std"] = -1
    
    return results


def dummy_generator(num_images: int, shape: Tuple[int, ...] = (3, 32, 32)) -> torch.Tensor:
    """Dummy generator for testing."""
    return torch.rand((num_images, *shape)) * 2 - 1


# =============================================================================
# Main (Testing)
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Metrics Utils on {device}")
    print("=" * 60)
    
    # Test real data preparation
    print("\n1. Preparing real data cache...")
    real_dir = save_cifar10_real_subset(num_images=100)
    
    # Test fake data generation
    print("\n2. Generating fake data...")
    fake_dir = save_fake_images(
        lambda n: dummy_generator(n),
        num_images=100,
        device=device
    )
    
    # Test individual metrics
    print("\n3. Testing individual metrics...")
    
    # FID
    try:
        fid = compute_fid(str(real_dir), str(fake_dir), device)
        print(f"  ✅ {fid}")
    except Exception as e:
        print(f"  ❌ FID: {e}")
    
    # IS
    try:
        fake_images = load_images_from_dir(str(fake_dir))
        is_score = compute_inception_score(fake_images, device)
        print(f"  ✅ {is_score}")
    except Exception as e:
        print(f"  ❌ IS: {e}")
    
    # Diversity
    try:
        extractor = InceptionFeatureExtractor(device)
        features = extractor.extract(fake_images)
        vendi = compute_vendi_score(features)
        print(f"  ✅ {vendi}")
    except Exception as e:
        print(f"  ❌ VENDI: {e}")
    
    # Full evaluation
    print("\n4. Running full evaluation...")
    evaluator = DiffusionEvaluator(device=device, real_data_dir=str(real_dir))
    
    report = evaluator.evaluate(
        generator=lambda n: dummy_generator(n),
        num_samples=50,
        batch_size=16,
        strategy_name="DummyTest",
        model_name="DummyModel"
    )
    
    report.print_summary()
    report.save_json("test_report.json")
    
    print("\n✅ All tests completed!")
