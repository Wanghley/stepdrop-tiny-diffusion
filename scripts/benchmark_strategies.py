import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import numpy as np

# Adjust path to allow imports from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.step_skipper import StochasticStepSkipScheduler, AdaptiveStepSkipScheduler
from scripts.ddpm_sampler import DDPMSampler
from scripts.ddim_sampler import DDIMSampler
from src.eval import metrics_utils

#usage: python benchmark_strategies.py --dummy --samples 100
#replace with: python benchmark_strategies.py --checkpoint_path path/to/your/checkpoint

class BenchmarkRunner:
    """
    Orchestrates the benchmarking of different diffusion sampling strategies.
    
    Metrics collected:
    - Wall-clock latency (s)
    - Throughput (img/s)
    - FID (Fréchet Inception Distance)
    - IS (Inception Score)
    - Total FLOPs (estimated)
    """
    
    def __init__(self, output_dir: str = "results", device: str = "cuda"):
        self.output_dir = Path(output_dir)
        self.device = device
        self.results = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, checkpoint_path: str = None, dummy: bool = False) -> nn.Module:
        """Loads the real model or a dummy model for testing."""
        if dummy:
            print("⚠️ Running in DUMMY mode with MockModel")
            class MockModel(nn.Module):
                def forward(self, x, t):
                    # Simulate some compute
                    return torch.randn_like(x) * 0.1
            return MockModel().to(self.device)
        
        if checkpoint_path is None:
            raise ValueError("Must provide checkpoint_path or set dummy=True")
            
        # TODO: Load real U-Net here once implemented
        print(f"Loading model from {checkpoint_path}...")
        # model = ...
        # model.load_state_dict(torch.load(checkpoint_path))
        raise NotImplementedError("Real model loading not yet implemented")

    def run_strategy(
        self, 
        name: str, 
        model: nn.Module, 
        sampler_config: Dict[str, Any],
        num_samples: int = 100,
        batch_size: int = 16,
        real_data_dir: str = "data/fid_real_cifar10"
    ):
        """
        Runs a single benchmarking strategy.
        """
        print(f"\n🚀 Running Strategy: {name}")
        strategy_dir = self.output_dir / name
        strategy_dir.mkdir(exist_ok=True)
        
        # 1. Select Sampler
        stype = sampler_config.pop("type")
        if stype == "ddpm":
            sampler = DDPMSampler(num_timesteps=1000)
            sample_fn = lambda x, shape: sampler.sample(model, shape, device=self.device)
            steps_per_iso = 1000
        elif stype == "ddim":
            steps = sampler_config.get("num_inference_steps", 50)
            sampler = DDIMSampler(num_timesteps=1000)
            sample_fn = lambda x, shape: sampler.sample(
                model, shape, device=self.device, **sampler_config
            )
            steps_per_iso = steps
        elif stype == "stepdrop":
            sampler = StochasticStepSkipScheduler(num_timesteps=1000)
            sample_fn = lambda x, shape: sampler.sample(
                model, shape, device=self.device, return_stats=True, **sampler_config
            )
            # Steps will vary, we'll get average from stats
            steps_per_iso = None 
        else:
            raise ValueError(f"Unknown sampler type: {stype}")

        # 2. Measure FLOPs (per step)
        # We assume 3x32x32 input for CIFAR
        single_step_flops = metrics_utils.computeFLOPs(model, (1, 3, 32, 32), device=self.device)
        
        # 3. Generate Samples & Measure Latency
        print(f"   Generating {num_samples} samples...")
        start_time = time.time()
        
        generated_count = 0
        total_steps_executed = 0
        
        # Wrapper for saveFakeImages
        def generator_wrapper(n_imgs):
            nonlocal generated_count, total_steps_executed
            # We need to reshape slightly because saveFakeImages calls this in a loop
            # But our sample_fn handles batching logic if we just pass shape
            
            # Since saveFakeImages asks for n_imgs, we perform one batch of size n_imgs
            # Note: simplistic handling, assuming n_imgs fits in VRAM for this bench
            current_batch = n_imgs
            
            # Handle return_stats for StepDrop
            result = sample_fn(None, (current_batch, 3, 32, 32))
            
            if isinstance(result, tuple):
                imgs, stats = result
                total_steps_executed += stats["steps_executed"]
            else:
                imgs = result
                total_steps_executed += steps_per_iso * current_batch
                
            generated_count += current_batch
            return imgs
            
        # Run generation via the utility which saves images
        # We'll use a larger batch size for the metric utility's loop
        fake_images_dir = metrics_utils.saveFakeImages(
            generator_wrapper,
            outDir=str(strategy_dir / "samples"),
            numImages=num_samples,
            batchSize=batch_size,
            device=self.device
        )
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = num_samples / duration
        avg_steps = total_steps_executed / num_samples
        total_flops = single_step_flops * avg_steps
        
        print(f"   ⏱️ Duration: {duration:.2f}s ({throughput:.2f} img/s)")
        print(f"   🔢 Avg Steps: {avg_steps:.1f}")
        
        # 4. Compute Metrics (FID, IS)
        print("   Computing Metrics...")
        try:
            fid = metrics_utils.computeFID(real_data_dir, str(fake_images_dir), device=self.device)
            is_mean, is_std = metrics_utils.computeInceptionScore(str(fake_images_dir), device=self.device)
        except Exception as e:
            print(f"   ⚠️ Metric calculation failed: {e}")
            fid, is_mean, is_std = -1, -1, -1

        print(f"   📉 FID: {fid:.2f}")
        print(f"   📈 IS: {is_mean:.2f}")
        
        # 5. Store Results
        self.results[name] = {
            "duration_sec": duration,
            "throughput": throughput,
            "avg_steps": avg_steps,
            "fid": fid,
            "is_mean": is_mean,
            "is_std": is_std,
            "flops_per_sample": total_flops,
            "config": sampler_config
        }
        self.save_report()

    def save_report(self):
        report_path = self.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"\n📝 Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description="""
StepDrop Benchmarking Harness
=============================

Runs a comprehensive evaluation of different diffusion sampling strategies.
Metrics collected: FID, Inception Score, FLOPs, Latency.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  1. Quick Dry-Run (Dummy Model):
     $ python scripts/benchmark_strategies.py --dummy --samples 10

  2. Full Evaluation (Real Model):
     $ python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 5000

  3. Custom Batch Size:
     $ python scripts/benchmark_strategies.py --dummy --batch_size 32

OUTPUT:
  Results are saved to: results/YYYY-MM-DD_HH-MM-SS/
  Contains:
    - report.json (All metrics)
    - */samples/*.png (Generated images)
"""
    )
    parser.add_argument("--dummy", action="store_true", help="Run with dummy model (no GPU required)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples per strategy to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner = BenchmarkRunner(output_dir=f"results/{timestamp}", device=device)
    
    model = runner.load_model(args.checkpoint, dummy=args.dummy)
    
    # Define Strategies
    strategies = [
        {
            "name": "DDPM_Baseline",
            "config": {"type": "ddpm"}
        },
        {
            "name": "DDIM_50",
            "config": {"type": "ddim", "num_inference_steps": 50, "eta": 0.0}
        },
        {
            "name": "StepDrop_Linear_0.3",
            "config": {"type": "stepdrop", "skip_strategy": "linear", "base_skip_prob": 0.3}
        },
        {
            "name": "StepDrop_Adaptive_0.2",
            "config": {"type": "stepdrop", "skip_strategy": "adaptive", "base_skip_prob": 0.2}
        }
    ]
    
    # Ensure real data cache exists (needed for FID)
    # We use a small subset for dummy runs to speed it up
    num_real = 100 if args.dummy else 5000
    metrics_utils.saveCifar10RealSub(numImages=num_real)
    
    for strategy in strategies:
        runner.run_strategy(
            strategy["name"],
            model,
            strategy["config"],
            num_samples=args.samples,
            batch_size=args.batch_size,
            real_data_dir="data/fid_real_cifar10"
        )
        
    runner.save_report()

if __name__ == "__main__":
    main()
