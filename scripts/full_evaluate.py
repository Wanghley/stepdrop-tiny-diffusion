#!/usr/bin/env python3
"""
Comprehensive Evaluation Script
===============================

Full evaluation using all metrics from metrics_utils.py

Metrics:
- FID (Fréchet Inception Distance)
- Clean-FID (more consistent FID)
- KID (Kernel Inception Distance)
- IS (Inception Score)
- Precision & Recall
- Density & Coverage
- LPIPS (Perceptual similarity)
- SSIM / MS-SSIM
- PSNR
- VENDI Score (diversity)
- Intra-LPIPS (diversity)
- FLOPs / Throughput / Memory

Usage:
    python scripts/full_evaluate.py --real_dir data/fid_real_cifar10 --fake_dir samples/ddpm
    python scripts/full_evaluate.py --real_dir data/fid_real_cifar10 --fake_dir samples/stepdrop --output results/eval.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics_utils import (
    MetricResult,
    EvaluationReport,
    compute_fid,
    compute_clean_fid,
    compute_kid,
    compute_inception_score,
    compute_precision_recall,
    compute_density_coverage,
    compute_lpips,
    compute_ssim,
    compute_psnr,
    compute_ms_ssim,
    compute_vendi_score,
    compute_intra_lpips,
    compute_throughput,
    compute_memory_usage,
    load_images_from_dir,
    HAS_PYTORCH_FID,
    HAS_CLEAN_FID,
    HAS_TORCHMETRICS,
    HAS_LPIPS,
    HAS_PRDC,
    HAS_FVCORE,
)


def check_dependencies():
    """Print dependency status."""
    print("\n📦 Dependency Status:")
    deps = [
        ("pytorch-fid", HAS_PYTORCH_FID, "FID computation"),
        ("clean-fid", HAS_CLEAN_FID, "Clean-FID, KID computation"),
        ("torchmetrics", HAS_TORCHMETRICS, "IS, SSIM, PSNR, MS-SSIM"),
        ("lpips", HAS_LPIPS, "LPIPS, Intra-LPIPS"),
        ("prdc", HAS_PRDC, "Precision/Recall, Density/Coverage"),
        ("fvcore", HAS_FVCORE, "FLOPs computation"),
    ]
    
    for name, available, purpose in deps:
        status = "✓" if available else "✗"
        color = "" if available else " (pip install " + name + ")"
        print(f"  {status} {name}: {purpose}{color}")
    print()


def run_full_evaluation(
    real_dir: str,
    fake_dir: str,
    device: str = "cuda",
    compute_diversity: bool = True,
    num_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Run all available metrics on real vs fake images.
    
    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing generated images
        device: Device to use
        compute_diversity: Whether to compute diversity metrics (slower)
        num_samples: Max samples for some metrics
    
    Returns:
        Dictionary of metric results
    """
    results = {}
    
    # === Quality Metrics ===
    print("\n" + "="*60)
    print("📊 QUALITY METRICS")
    print("="*60)
    
    # FID
    print("\n📈 Computing FID...")
    fid_result = compute_fid(real_dir, fake_dir, device=device)
    results["fid"] = fid_result.value
    print(f"   FID: {fid_result.value:.4f}")
    
    # Clean-FID
    if HAS_CLEAN_FID:
        print("\n📈 Computing Clean-FID...")
        cfid_result = compute_clean_fid(real_dir, fake_dir, device=device)
        results["clean_fid"] = cfid_result.value
        print(f"   Clean-FID: {cfid_result.value:.4f}")
    
    # KID
    if HAS_CLEAN_FID:
        print("\n📈 Computing KID...")
        kid_result = compute_kid(real_dir, fake_dir, device=device)
        results["kid"] = kid_result.value
        results["kid_std"] = kid_result.std
        print(f"   KID: {kid_result.value:.6f} ± {kid_result.std:.6f}")
    
    # Inception Score
    print("\n📈 Computing Inception Score...")
    is_result = compute_inception_score(fake_dir, device=device)
    results["is_mean"] = is_result.value
    results["is_std"] = is_result.std
    print(f"   IS: {is_result.value:.4f} ± {is_result.std:.4f}")
    
    # Precision/Recall
    if HAS_PRDC:
        print("\n📈 Computing Precision/Recall...")
        pr_result = compute_precision_recall(real_dir, fake_dir, device=device)
        if pr_result.extra:
            results["precision"] = pr_result.extra.get("precision", -1)
            results["recall"] = pr_result.extra.get("recall", -1)
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall: {results['recall']:.4f}")
    
    # Density/Coverage
    if HAS_PRDC:
        print("\n📈 Computing Density/Coverage...")
        dc_result = compute_density_coverage(real_dir, fake_dir, device=device)
        if dc_result.extra:
            results["density"] = dc_result.extra.get("density", -1)
            results["coverage"] = dc_result.extra.get("coverage", -1)
            print(f"   Density: {results['density']:.4f}")
            print(f"   Coverage: {results['coverage']:.4f}")
    
    # === Perceptual Metrics ===
    print("\n" + "="*60)
    print("👁️ PERCEPTUAL METRICS")
    print("="*60)
    
    # LPIPS
    if HAS_LPIPS:
        print("\n📈 Computing LPIPS...")
        lpips_result = compute_lpips(real_dir, fake_dir, device=device, num_samples=num_samples)
        results["lpips"] = lpips_result.value
        results["lpips_std"] = lpips_result.std
        print(f"   LPIPS: {lpips_result.value:.4f} ± {lpips_result.std:.4f}")
    
    # SSIM
    print("\n📈 Computing SSIM...")
    ssim_result = compute_ssim(real_dir, fake_dir, device=device, num_samples=num_samples)
    results["ssim"] = ssim_result.value
    results["ssim_std"] = ssim_result.std
    print(f"   SSIM: {ssim_result.value:.4f} ± {ssim_result.std:.4f}")
    
    # PSNR
    print("\n📈 Computing PSNR...")
    psnr_result = compute_psnr(real_dir, fake_dir, device=device, num_samples=num_samples)
    results["psnr"] = psnr_result.value
    results["psnr_std"] = psnr_result.std
    print(f"   PSNR: {psnr_result.value:.4f} ± {psnr_result.std:.4f}")
    
    # MS-SSIM
    if HAS_TORCHMETRICS:
        print("\n📈 Computing MS-SSIM...")
        msssim_result = compute_ms_ssim(real_dir, fake_dir, device=device, num_samples=num_samples)
        results["ms_ssim"] = msssim_result.value
        results["ms_ssim_std"] = msssim_result.std
        print(f"   MS-SSIM: {msssim_result.value:.4f} ± {msssim_result.std:.4f}")
    
    # === Diversity Metrics ===
    if compute_diversity:
        print("\n" + "="*60)
        print("🎨 DIVERSITY METRICS")
        print("="*60)
        
        # VENDI Score
        print("\n📈 Computing VENDI Score...")
        vendi_result = compute_vendi_score(fake_dir, device=device, num_samples=min(500, num_samples))
        results["vendi"] = vendi_result.value
        print(f"   VENDI: {vendi_result.value:.4f}")
        
        # Intra-LPIPS
        if HAS_LPIPS:
            print("\n📈 Computing Intra-LPIPS...")
            ilpips_result = compute_intra_lpips(fake_dir, device=device, num_samples=min(500, num_samples))
            results["intra_lpips"] = ilpips_result.value
            results["intra_lpips_std"] = ilpips_result.std
            print(f"   Intra-LPIPS: {ilpips_result.value:.4f} ± {ilpips_result.std:.4f}")
    
    # === Summary ===
    print("\n" + "="*60)
    print("📋 EVALUATION SUMMARY")
    print("="*60)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive diffusion model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic evaluation
  python scripts/full_evaluate.py --real_dir data/fid_real_cifar10 --fake_dir samples/

  # With output file
  python scripts/full_evaluate.py --real_dir data/real --fake_dir samples/ddpm --output results.json

  # Skip diversity metrics (faster)
  python scripts/full_evaluate.py --real_dir data/real --fake_dir samples/ --no_diversity
"""
    )
    
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory containing real images")
    parser.add_argument("--fake_dir", type=str, required=True,
                        help="Directory containing generated/fake images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Max samples for pairwise metrics")
    parser.add_argument("--no_diversity", action="store_true",
                        help="Skip diversity metrics (faster)")
    parser.add_argument("--check_deps", action="store_true",
                        help="Only check dependencies and exit")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    if args.check_deps:
        return
    
    # Validate directories
    real_path = Path(args.real_dir)
    fake_path = Path(args.fake_dir)
    
    if not real_path.exists():
        print(f"❌ Real directory not found: {real_path}")
        sys.exit(1)
    
    if not fake_path.exists():
        print(f"❌ Fake directory not found: {fake_path}")
        sys.exit(1)
    
    # Count images
    real_count = len(list(real_path.glob("*.png"))) + len(list(real_path.glob("*.jpg")))
    fake_count = len(list(fake_path.glob("*.png"))) + len(list(fake_path.glob("*.jpg")))
    
    print(f"📁 Real images: {real_count}")
    print(f"📁 Fake images: {fake_count}")
    
    if real_count == 0 or fake_count == 0:
        print("❌ Need at least some images in both directories")
        sys.exit(1)
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = "cpu"
    
    print(f"🖥️ Device: {device}")
    
    # Run evaluation
    results = run_full_evaluation(
        real_dir=str(real_path),
        fake_dir=str(fake_path),
        device=device,
        compute_diversity=not args.no_diversity,
        num_samples=args.num_samples,
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "real_dir": str(real_path),
            "fake_dir": str(fake_path),
            "device": device,
            "metrics": results
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
