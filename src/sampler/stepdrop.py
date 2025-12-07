"""
StepDrop Sampler
================

Stochastic Step Skipping for accelerated diffusion sampling.

StepDrop introduces controlled randomness in which timesteps are evaluated,
providing a tunable tradeoff between speed and quality.

Two variants are provided:
- StepDropSampler: Fixed probability step skipping
- AdaptiveStepDropSampler: Error-based adaptive step skipping
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class StepDropStats:
    """Statistics from a StepDrop sampling run."""
    total_timesteps: int
    steps_taken: int
    steps_skipped: int
    skip_rate: float
    timesteps_used: list
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_timesteps": self.total_timesteps,
            "steps_taken": self.steps_taken,
            "steps_skipped": self.steps_skipped,
            "skip_rate": self.skip_rate,
            "timesteps_used": self.timesteps_used
        }


class StepDropSampler:
    """
    StepDrop Sampler with Stochastic Step Skipping.
    
    At each timestep, randomly decides whether to skip based on a probability
    that can vary across the diffusion process.
    
    Args:
        num_timesteps: Total number of diffusion timesteps (default: 1000)
        beta_schedule: Type of beta schedule ("cosine" or "linear")
    
    Skip Strategies:
        - "linear": Skip probability increases linearly from 0 at ends to peak at middle
        - "cosine_sq": Skip probability follows cos²(πt) curve - smooth, peaks in middle
        - "quadratic": Skip probability follows quadratic curve - sharper peak in middle
        - "constant": Fixed skip probability throughout (baseline)
    
    Example:
        >>> sampler = StepDropSampler(num_timesteps=1000)
        >>> samples, stats = sampler.sample(
        ...     model, shape, skip_prob=0.3, skip_strategy="linear"
        ... )
        >>> print(f"Steps taken: {stats.steps_taken}")
    """
    
    def __init__(
        self, 
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        
        # Compute beta schedule
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), 
            self.alphas_cumprod[:-1]
        ])
        
        # Posterior coefficients (same as DDPM)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def _get_skip_probability(
        self, 
        t: int, 
        base_prob: float,
        strategy: str
    ) -> float:
        """
        Compute skip probability for timestep t.
        
        Args:
            t: Current timestep (0 to num_timesteps-1)
            base_prob: Base skip probability
            strategy: Skip strategy name
        
        Returns:
            Probability of skipping this timestep
        """
        # Normalize timestep to [0, 1]
        # t=0 is final step (clean), t=num_timesteps-1 is first step (noisy)
        t_norm = t / (self.num_timesteps - 1)
        
        if strategy == "constant":
            # Constant skip probability (baseline)
            return base_prob
        
        elif strategy == "linear":
            # Linear: peaks at middle (t_norm=0.5), zero at ends
            # p(t) = base_prob * 4 * t * (1-t)
            # This is actually parabolic but commonly called "linear" schedule
            return base_prob * 4 * t_norm * (1 - t_norm)
        
        elif strategy == "cosine_sq" or strategy == "cosine":
            # Cosine²: smooth curve, peaks at middle
            # p(t) = base_prob * sin²(π * t) = base_prob * (1 - cos(2πt)) / 2
            # Smoother transitions than linear, gentler at boundaries
            return base_prob * (np.sin(np.pi * t_norm) ** 2)
        
        elif strategy == "quadratic":
            # Quadratic: sharper peak in middle than linear
            # p(t) = base_prob * 16 * t² * (1-t)²
            # More aggressive skipping in the middle, more conservative at ends
            return base_prob * 16 * (t_norm ** 2) * ((1 - t_norm) ** 2)
        
        elif strategy == "early_skip":
            # Higher skip probability for early (high t) timesteps
            # These are high-noise, coarse structure steps
            return base_prob * t_norm
        
        elif strategy == "late_skip":
            # Higher skip probability for late (low t) timesteps
            # These are low-noise, fine detail steps
            return base_prob * (1 - t_norm)
        
        elif strategy == "critical_preserve":
            # Preserve critical middle timesteps, skip at extremes
            # Critical region: t_norm in [0.3, 0.7]
            if 0.3 <= t_norm <= 0.7:
                return base_prob * 0.1  # Very low skip in critical region
            else:
                return base_prob * 1.5  # Higher skip at extremes
        
        else:
            raise ValueError(f"Unknown skip strategy: {strategy}")
    
    def _should_skip(self, t: int, base_prob: float, strategy: str) -> bool:
        """Determine whether to skip timestep t."""
        # Never skip first or last few timesteps (critical for quality)
        if t >= self.num_timesteps - 5 or t <= 5:
            return False
        
        prob = self._get_skip_probability(t, base_prob, strategy)
        return torch.rand(1).item() < prob
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: str = "cuda",
        skip_prob: float = 0.3,
        skip_strategy: str = "linear",
        return_stats: bool = True,
        show_progress: bool = True,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, Optional[StepDropStats]]:
        """
        Generate samples using StepDrop.
        
        Args:
            model: Noise prediction model
            shape: Output shape (batch_size, channels, height, width)
            device: Device to run sampling on
            skip_prob: Base probability of skipping a timestep (0-1)
            skip_strategy: Strategy for varying skip probability
            return_stats: If True, return sampling statistics
            show_progress: If True, show progress bar
            clip_denoised: If True, clip predicted x_0 to [-1, 1]
        
        Returns:
            Tuple of (samples, stats) if return_stats=True, else just samples
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        timesteps_used = []
        steps_skipped = 0
        
        # Track last prediction for interpolation when skipping
        last_pred_x0 = None
        last_t = self.num_timesteps
        
        timesteps = list(reversed(range(self.num_timesteps)))
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"StepDrop ({skip_strategy})")
        
        for t in timesteps:
            # Decide whether to skip
            if self._should_skip(t, skip_prob, skip_strategy) and last_pred_x0 is not None:
                steps_skipped += 1
                continue
            
            timesteps_used.append(t)
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Predict x_0
            alpha_cumprod = self.alphas_cumprod[t].to(device)
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            last_pred_x0 = pred_x0
            last_t = t
            
            # Compute posterior mean
            coef1 = self.posterior_mean_coef1[t].to(device)
            coef2 = self.posterior_mean_coef2[t].to(device)
            mean = coef1 * pred_x0 + coef2 * x
            
            # Sample x_{t-1}
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[t].to(device)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        
        stats = None
        if return_stats:
            stats = StepDropStats(
                total_timesteps=self.num_timesteps,
                steps_taken=len(timesteps_used),
                steps_skipped=steps_skipped,
                skip_rate=steps_skipped / self.num_timesteps,
                timesteps_used=timesteps_used
            )
        
        return x, stats
    
    def get_expected_nfe(self, skip_prob: float, skip_strategy: str) -> int:
        """Estimate expected number of function evaluations."""
        # Account for protected timesteps (first and last 5)
        protected = 10
        skippable = self.num_timesteps - protected
        
        if skip_strategy == "constant":
            expected_skips = skippable * skip_prob
        elif skip_strategy in ["linear", "cosine"]:
            # Average skip prob is roughly base_prob * 0.5 for these
            expected_skips = skippable * skip_prob * 0.5
        else:
            expected_skips = skippable * skip_prob * 0.5
        
        return int(self.num_timesteps - expected_skips)


class AdaptiveStepDropSampler(StepDropSampler):
    """
    Adaptive StepDrop Sampler with Error-Based Step Skipping.
    
    Dynamically adjusts skip probability based on estimated reconstruction
    error. High error → take more steps. Low error → skip more.
    
    Args:
        num_timesteps: Total number of diffusion timesteps
        error_threshold_low: Below this error, increase skipping
        error_threshold_high: Above this error, decrease skipping
    
    Example:
        >>> sampler = AdaptiveStepDropSampler(num_timesteps=1000)
        >>> samples, stats = sampler.sample(model, shape, base_skip_prob=0.2)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        error_threshold_low: float = 0.05,
        error_threshold_high: float = 0.15,
        **kwargs
    ):
        super().__init__(num_timesteps=num_timesteps, **kwargs)
        self.error_threshold_low = error_threshold_low
        self.error_threshold_high = error_threshold_high
    
    def _estimate_error(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        t: torch.Tensor,
        noise_scale: float = 0.01
    ) -> float:
        """
        Estimate reconstruction error using prediction stability.
        
        Adds small perturbation and measures prediction difference.
        High difference → high uncertainty → don't skip.
        """
        with torch.no_grad():
            pred_noise_1 = model(x, t)
            
            # Add small perturbation
            perturbation = torch.randn_like(x) * noise_scale
            pred_noise_2 = model(x + perturbation, t)
            
            # Measure prediction stability
            error = (pred_noise_1 - pred_noise_2).abs().mean().item()
        
        return error
    
    def _compute_adaptive_skip_prob(
        self,
        error: float,
        base_prob: float
    ) -> float:
        """
        Compute skip probability based on estimated error.
        
        Low error → high skip probability (can afford to skip)
        High error → low skip probability (need careful denoising)
        """
        if error < self.error_threshold_low:
            # Low error: can skip more
            return min(base_prob * 2, 0.8)
        elif error > self.error_threshold_high:
            # High error: skip less
            return base_prob * 0.25
        else:
            # Interpolate
            error_norm = (error - self.error_threshold_low) / (
                self.error_threshold_high - self.error_threshold_low
            )
            return base_prob * (1.5 - error_norm)
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: str = "cuda",
        base_skip_prob: float = 0.2,
        quality_threshold: float = -0.1,  # Unused, kept for API compatibility
        return_stats: bool = True,
        show_progress: bool = True,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Generate samples using Adaptive StepDrop.
        
        Args:
            model: Noise prediction model
            shape: Output shape
            device: Device to run on
            base_skip_prob: Base skip probability (adjusted by error)
            quality_threshold: Deprecated, kept for compatibility
            return_stats: If True, return statistics
            show_progress: If True, show progress bar
            clip_denoised: If True, clip predicted x_0
        
        Returns:
            Tuple of (samples, stats_dict)
        """
        batch_size = shape[0]
        
        x = torch.randn(shape, device=device)
        
        timesteps_used = []
        errors_recorded = []
        skip_probs_used = []
        
        timesteps = list(reversed(range(self.num_timesteps)))
        if show_progress:
            timesteps = tqdm(timesteps, desc="Adaptive StepDrop")
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Estimate error
            error = self._estimate_error(model, x, t_batch)
            errors_recorded.append(error)
            
            # Compute adaptive skip probability
            skip_prob = self._compute_adaptive_skip_prob(error, base_skip_prob)
            skip_probs_used.append(skip_prob)
            
            # Decide whether to skip (never skip protected timesteps)
            if t >= self.num_timesteps - 5 or t <= 5:
                should_skip = False
            else:
                should_skip = torch.rand(1).item() < skip_prob
            
            if should_skip:
                continue
            
            timesteps_used.append(t)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Predict x_0
            alpha_cumprod = self.alphas_cumprod[t].to(device)
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Compute posterior mean
            coef1 = self.posterior_mean_coef1[t].to(device)
            coef2 = self.posterior_mean_coef2[t].to(device)
            mean = coef1 * pred_x0 + coef2 * x
            
            # Sample
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[t].to(device)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        
        stats = None
        if return_stats:
            stats = {
                "total_timesteps": self.num_timesteps,
                "steps_taken": len(timesteps_used),
                "steps_skipped": self.num_timesteps - len(timesteps_used),
                "skip_rate": (self.num_timesteps - len(timesteps_used)) / self.num_timesteps,
                "timesteps_used": timesteps_used,
                "errors": errors_recorded,
                "skip_probs": skip_probs_used,
                "mean_error": np.mean(errors_recorded),
                "mean_skip_prob": np.mean(skip_probs_used)
            }
        
        return x, stats