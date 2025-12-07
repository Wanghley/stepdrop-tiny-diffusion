"""
Diffusion Model Samplers
========================

This module provides various sampling strategies for diffusion models:

- DDPMSampler: Standard DDPM sampling (1000 steps)
- DDIMSampler: Accelerated DDIM sampling (configurable steps)
- StepDropSampler: Stochastic step skipping for faster sampling
- AdaptiveStepDropSampler: Adaptive step skipping based on error estimation

Usage:
    from src.sampler import DDPMSampler, DDIMSampler, StepDropSampler

    # DDPM (slow but high quality)
    sampler = DDPMSampler(num_timesteps=1000)
    samples = sampler.sample(model, shape, device="cuda")

    # DDIM (fast)
    sampler = DDIMSampler(num_timesteps=1000, num_inference_steps=50)
    samples = sampler.sample(model, shape, device="cuda")

    # StepDrop (adaptive speed/quality tradeoff)
    sampler = StepDropSampler(num_timesteps=1000)
    samples, stats = sampler.sample(model, shape, skip_prob=0.3, device="cuda")
"""

from .DDPM import DDPMSampler
from .DDIM import DDIMSampler
from .stepdrop import StepDropSampler, AdaptiveStepDropSampler

__all__ = [
    "DDPMSampler",
    "DDIMSampler", 
    "StepDropSampler",
    "AdaptiveStepDropSampler",
]