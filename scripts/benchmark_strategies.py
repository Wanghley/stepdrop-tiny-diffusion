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
    
    def __init__(self, output_dir: str = "results", device: str = "cuda", real_data_path: str = "data/fid_real_cifar10"):
        self.output_dir = Path(output_dir)
        self.device = device
        self.results = {}
        self.real_data_path = real_data_path
        
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

    def run_strategy(self, name: str, model: nn.Module, sampler_config: Dict[str, Any], num_samples: int, batch_size: int = 100):
        print(f"\n🚀 Running Strategy: {name}")
        
        # Initialize Sampler
        if "DDPM" in name:
            sampler = DDPMSampler(num_timesteps=1000)
            sampler_args = {
                "device": self.device,
                "return_all_timesteps": False
            }
        else:
            # DDIM and StepDrop now both use DDIMSampler
            sampler = DDIMSampler(num_timesteps=1000)
            
            # Default DDIM args
            sampler_args = {
                "num_inference_steps": sampler_config.get("num_inference_steps", 50),
                "eta": sampler_config.get("eta", 0.0),
                "device": self.device,
                "return_all_timesteps": False,
                "schedule_type": sampler_config.get("schedule_type", "uniform")
            }
            
            # Handle Adaptive (StepDrop) params
            if sampler_args["schedule_type"] == "adaptive":
                sampler_args["adaptive_params"] = {
                    "min_step": sampler_config.get("min_step", 5),
                    "max_step": sampler_config.get("max_step", 50),
                    "error_threshold_low": sampler_config.get("error_threshold_low", 0.05),
                    "error_threshold_high": sampler_config.get("error_threshold_high", 0.15),
                }

        # 1. Compute FLOPs (Theoretical)
        # For StepDrop, we ideally want average actual steps, but theoretical FLOPs 
        # is usually calculated per pure model call. 
        # We will track *actual* model calls during sampling.
        single_step_flops = metrics_utils.computeFLOPs(model, (1, 3, 32, 32), self.device)
        
        # 2. Generate Samples & Measure Latency
        print(f"   Generating {num_samples} samples...")
        start_time = time.time()
        
        # We need to batch generation to avoid OOM
        # batch_size arg is used here
        all_samples = []
        total_steps_executed = 0
        
        # Determine how many batches
        n_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in metrics_utils.tqdm(range(n_batches), desc="Generating batches"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            shape = (current_batch_size, 3, 32, 32)
            
            with torch.no_grad():
                result = sampler.sample(model, shape, **sampler_args)
            
            # Handle different return types (DDPM returns tensor, DDIM returns dict)
            if isinstance(result, dict):
                batch_samples = result['samples']
                # Count steps for FLOPs calculation
                if 'timesteps_used' in result:
                    total_steps_executed += len(result['timesteps_used']) * current_batch_size
                else:
                    # Fallback if specific steps not returned, assume max
                    total_steps_executed += sampler_args.get("num_inference_steps", 50) * current_batch_size
            else:
                batch_samples = result
                # DDPM always runs full steps
                total_steps_executed += 1000 * current_batch_size
                
            all_samples.append(batch_samples.cpu())
            
        duration = time.time() - start_time
        fps = num_samples / duration
        
        # Concatenate all batches
        samples = torch.cat(all_samples, dim=0)
        
        # Normalize and Save Images
        # Assuming model output is [-1, 1], normalize to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Save samples to disk
        strategy_dir = self.output_dir / name
        strategy_dir.mkdir(exist_ok=True)
        
        # Clear existing images
        for f in strategy_dir.glob("*.png"):
            f.unlink()
            
        # Save images
        for idx, img in enumerate(samples):
            # img is [C, H, W] in [0, 1]
            from torchvision.utils import save_image
            save_image(img, strategy_dir / f"{idx:06d}.png")
        
        # 3. Calculate Metrics
        avg_steps = total_steps_executed / num_samples
        total_flops = single_step_flops * avg_steps
        
        print(f"   ⏱️ Duration: {duration:.2f}s ({fps:.2f} img/s)")
        print(f"   🔢 Avg Steps: {avg_steps:.1f}")
        
        print("   Computing Metrics...")
        fid = metrics_utils.computeFID(self.real_data_path, str(strategy_dir), self.device)
        is_mean, is_std = metrics_utils.computeInceptionScore(str(strategy_dir), self.device)
        
        print(f"   📉 FID: {fid:.2f}")
        print(f"   📈 IS: {is_mean:.2f}")
        
        # Store results
        self.results[name] = {
            "duration": duration,
            "throughput": fps,
            "fid": fid,
            "is_mean": is_mean,
            "is_std": is_std,
            "avg_steps": avg_steps,
            "flops_per_sample": total_flops
        }
        
        # Save incremental report
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
            "config": {}
        },
        {
            "name": "DDIM_Uniform_50",
            "config": {"num_inference_steps": 50, "eta": 0.0, "schedule_type": "uniform"}
        },
        {
            "name": "DDIM_Cosine_50",
            "config": {"num_inference_steps": 50, "eta": 0.0, "schedule_type": "cosine"}
        },
        {
            "name": "StepDrop_Adaptive",
            "config": {
                "schedule_type": "adaptive", 
                "min_step": 5, 
                "max_step": 50,
                "error_threshold_low": 0.05,
                "error_threshold_high": 0.15
            }
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
            batch_size=args.batch_size
        )
        
    runner.save_report()

if __name__ == "__main__":
    main()
