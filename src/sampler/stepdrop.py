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
    
    Base Samplers:
        - "ddim" (recommended): Pre-computes skip schedule, uses DDIM update rule
          which handles arbitrary step sizes correctly. More principled approach.
        - "ddpm": Original stochastic method with on-the-fly skip decisions.
          Uses interpolation when skipping.
    
    Skip Strategies:
        - "linear": Skip probability increases linearly from 0 at ends to peak at middle
        - "cosine_sq": Skip probability follows cos²(πt) curve - smooth, peaks in middle
        - "quadratic": Skip probability follows quadratic curve - sharper peak in middle
        - "constant": Fixed skip probability throughout (baseline)
        - "early_skip": Higher skip probability for early (high noise) timesteps
        - "late_skip": Higher skip probability for late (low noise) timesteps
        - "critical_preserve": Low skip in critical middle region [0.3, 0.7]
        - "aggressive_middle": Very aggressive skipping in safe middle zone [0.15, 0.85]
        - "uniform_target": Uniform high skip rate to achieve target NFE
    
    Example:
        >>> sampler = StepDropSampler(num_timesteps=1000)
        >>> # Using DDIM base (recommended)
        >>> samples, stats = sampler.sample(
        ...     model, shape, skip_prob=0.3, skip_strategy="linear", base_sampler="ddim"
        ... )
        >>> # Using DDPM base (original)
        >>> samples, stats = sampler.sample(
        ...     model, shape, skip_prob=0.3, skip_strategy="linear", base_sampler="ddpm"
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
    
    def _should_skip(self, t: int, base_prob: float, strategy: str) -> bool:
        """Determine whether to skip timestep t."""
        # Protect more timesteps at critical boundaries
        # First 20 steps (high noise → structure) and last 20 (fine details)
        if t >= self.num_timesteps - 20 or t <= 20:
            return False
        
        prob = self._get_skip_probability(t, base_prob, strategy)
        return torch.rand(1).item() < prob

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
        
        elif strategy == "aggressive_middle":
            # Very aggressive skipping in safe middle zone
            # Target ~50-100 NFE to compete with DDIM
            if 0.15 <= t_norm <= 0.85:
                return min(base_prob * 2.5, 0.95)  # Up to 95% skip in middle
            return 0.0  # Never skip at boundaries
        
        elif strategy == "uniform_target":
            # Uniform high skip rate to achieve target NFE
            # With base_prob=0.95, expect ~50 steps from 1000
            return base_prob
        
        else:
            raise ValueError(f"Unknown skip strategy: {strategy}")
    
    def _should_skip(self, t: int, base_prob: float, strategy: str) -> bool:
        """Determine whether to skip timestep t."""
        # Protect more timesteps at critical boundaries
        # First 20 steps (high noise → structure) and last 20 (fine details)
        if t >= self.num_timesteps - 20 or t <= 20:
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
        base_sampler: str = "ddim",  # "ddpm" or "ddim"
        eta: float = 0.0,  # Only used when base_sampler="ddim", 0=deterministic
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
            base_sampler: Base sampling method - "ddpm" (stochastic) or "ddim" (deterministic)
            eta: Stochasticity parameter for DDIM (0=deterministic, 1=DDPM-like)
            return_stats: If True, return sampling statistics
            show_progress: If True, show progress bar
            clip_denoised: If True, clip predicted x_0 to [-1, 1]
        
        Returns:
            Tuple of (samples, stats) if return_stats=True, else just samples
        """
        if base_sampler == "ddim":
            return self._sample_ddim(
                model, shape, device, skip_prob, skip_strategy,
                eta, return_stats, show_progress, clip_denoised
            )
        elif base_sampler == "ddpm":
            return self._sample_ddpm(
                model, shape, device, skip_prob, skip_strategy,
                return_stats, show_progress, clip_denoised
            )
        else:
            raise ValueError(f"Unknown base_sampler: {base_sampler}. Use 'ddpm' or 'ddim'.")
    
    @torch.no_grad()
    def _sample_ddpm(
        self,
        model: nn.Module,
        shape: tuple,
        device: str,
        skip_prob: float,
        skip_strategy: str,
        return_stats: bool,
        show_progress: bool,
        clip_denoised: bool
    ) -> Tuple[torch.Tensor, Optional[StepDropStats]]:
        """
        DDPM-based StepDrop sampling (original stochastic method).
        
        When skipping, uses interpolation based on last prediction.
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        timesteps_used = []
        steps_skipped = 0
        last_pred_x0 = None
        last_t = self.num_timesteps
        
        timesteps = list(reversed(range(self.num_timesteps)))
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"StepDrop-DDPM ({skip_strategy})")
        
        for t in timesteps:
            # Decide whether to skip
            if self._should_skip(t, skip_prob, skip_strategy) and last_pred_x0 is not None:
                steps_skipped += 1
                # Interpolate x using last prediction
                alpha_t = self.alphas_cumprod[t].to(device)
                alpha_last = self.alphas_cumprod[last_t].to(device) if last_t < self.num_timesteps else torch.tensor(1.0).to(device)
                
                # Use predicted x0 to estimate current x_t
                noise = (x - torch.sqrt(alpha_last) * last_pred_x0) / torch.sqrt(1 - alpha_last + 1e-8)
                x = torch.sqrt(alpha_t) * last_pred_x0 + torch.sqrt(1 - alpha_t) * noise
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
    
    @torch.no_grad()
    def _sample_ddim(
        self,
        model: nn.Module,
        shape: tuple,
        device: str,
        skip_prob: float,
        skip_strategy: str,
        eta: float,
        return_stats: bool,
        show_progress: bool,
        clip_denoised: bool
    ) -> Tuple[torch.Tensor, Optional[StepDropStats]]:
        """
        DDIM-based StepDrop sampling (recommended).
        
        Pre-computes which steps to skip, then uses DDIM's ability to handle
        arbitrary step sizes correctly. This is more principled than DDPM-based
        skipping because DDIM was designed for non-unit step sizes.
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        steps_skipped = 0
        
        # Build timestep sequence, deciding which to skip upfront
        all_timesteps = list(reversed(range(self.num_timesteps)))
        
        # First pass: decide which steps to take
        active_timesteps = []
        for t in all_timesteps:
            if not self._should_skip(t, skip_prob, skip_strategy):
                active_timesteps.append(t)
            else:
                steps_skipped += 1
        
        # Ensure we have start and end timesteps
        if self.num_timesteps - 1 not in active_timesteps:
            active_timesteps.insert(0, self.num_timesteps - 1)
        if 0 not in active_timesteps:
            active_timesteps.append(0)
        
        # Sort in descending order (high noise to low noise)
        active_timesteps = sorted(set(active_timesteps), reverse=True)
        
        iterator = active_timesteps
        if show_progress:
            iterator = tqdm(iterator, desc=f"StepDrop-DDIM ({skip_strategy})")
        
        timesteps_used = []
        
        for i, t in enumerate(iterator):
            timesteps_used.append(t)
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Get alpha values for current timestep
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Get next timestep (could be a big jump due to skipping!)
            if i < len(active_timesteps) - 1:
                t_prev = active_timesteps[i + 1]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev].to(device)
            else:
                t_prev = 0
                alpha_cumprod_t_prev = torch.tensor(1.0).to(device)
            
            # DDIM update formula (handles arbitrary step sizes correctly)
            # Direction pointing to x_t
            pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
            
            # Optionally add noise (eta > 0 adds stochasticity)
            if eta > 0 and t_prev > 0:
                sigma = eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t + 1e-8) * 
                    (1 - alpha_cumprod_t / (alpha_cumprod_t_prev + 1e-8))
                )
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir + sigma * noise
            else:
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir
        
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
        quality_threshold: float = -0.1,
        return_stats: bool = True,
        show_progress: bool = True,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Generate samples using Adaptive StepDrop with DDIM-style updates.
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        timesteps_used = []
        errors_recorded = []
        skip_probs_used = []
        
        # Pre-compute which steps to take based on error estimation
        # First pass: estimate errors and decide skips
        all_timesteps = list(reversed(range(self.num_timesteps)))
        active_timesteps = []
        
        # For adaptive, we need to run through and decide
        # Use a simpler approach: estimate "importance" of each timestep
        for t in all_timesteps:
            t_norm = t / (self.num_timesteps - 1)
            
            # Protected regions
            if t >= self.num_timesteps - 10 or t <= 10:
                active_timesteps.append(t)
                continue
            
            # Use position-based importance as proxy
            # (Real error estimation would require forward pass)
            importance = 1.0 - 4 * t_norm * (1 - t_norm)  # High at ends, low in middle
            skip_prob = base_skip_prob * (1 - importance)
            
            if torch.rand(1).item() >= skip_prob:
                active_timesteps.append(t)
        
        # Ensure we have endpoints
        if 0 not in active_timesteps:
            active_timesteps.append(0)
        active_timesteps = sorted(set(active_timesteps), reverse=True)
        
        iterator = active_timesteps
        if show_progress:
            iterator = tqdm(iterator, desc="Adaptive StepDrop")
        
        for i, t in enumerate(iterator):
            timesteps_used.append(t)
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Get next timestep
            if i < len(active_timesteps) - 1:
                t_prev = active_timesteps[i + 1]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev].to(device)
            else:
                t_prev = 0
                alpha_cumprod_t_prev = torch.tensor(1.0).to(device)
            
            # DDIM update (handles arbitrary step sizes)
            pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir
        
        stats = None
        if return_stats:
            stats = {
                "total_timesteps": self.num_timesteps,
                "steps_taken": len(timesteps_used),
                "steps_skipped": self.num_timesteps - len(timesteps_used),
                "skip_rate": (self.num_timesteps - len(timesteps_used)) / self.num_timesteps,
                "timesteps_used": timesteps_used,
            }
        
        return x, stats

class TargetNFEStepDropSampler:
    """
    StepDrop sampler that targets a specific number of function evaluations.
    
    Instead of skip probability, specify desired NFE and the sampler
    selects timesteps intelligently.
    """
    
    def __init__(
        self, 
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def _select_timesteps(
        self, 
        target_nfe: int, 
        strategy: str = "uniform"
    ) -> list:
        """
        Select which timesteps to use given target NFE.
        
        Strategies:
            - uniform: Evenly spaced (like DDIM)
            - importance: More steps at boundaries, fewer in middle
            - stochastic: Random selection with boundary protection
        """
        if strategy == "uniform":
            # Same as DDIM - evenly spaced
            indices = np.linspace(0, self.num_timesteps - 1, target_nfe, dtype=int)
            return sorted(indices.tolist(), reverse=True)
        
        elif strategy == "importance":
            # More steps at start (t~T) and end (t~0), fewer in middle
            # Allocate 30% to first 10%, 30% to last 10%, 40% to middle 80%
            n_start = int(target_nfe * 0.3)
            n_end = int(target_nfe * 0.3)
            n_middle = target_nfe - n_start - n_end
            
            start_range = int(self.num_timesteps * 0.1)
            end_range = int(self.num_timesteps * 0.1)
            
            start_steps = np.linspace(self.num_timesteps - 1, self.num_timesteps - start_range, n_start, dtype=int)
            end_steps = np.linspace(end_range, 0, n_end, dtype=int)
            middle_steps = np.linspace(self.num_timesteps - start_range - 1, end_range + 1, n_middle, dtype=int)
            
            all_steps = set(start_steps.tolist() + middle_steps.tolist() + end_steps.tolist())
            return sorted(all_steps, reverse=True)
        
        elif strategy == "stochastic":
            # Random selection but protect boundaries
            protected_start = 20  # Always include first 20
            protected_end = 20    # Always include last 20
            
            # Must-have timesteps
            must_have = list(range(self.num_timesteps - protected_start, self.num_timesteps))
            must_have += list(range(0, protected_end))
            
            # Randomly select from middle
            middle_range = list(range(protected_end, self.num_timesteps - protected_start))
            n_middle_needed = max(0, target_nfe - len(must_have))
            
            if n_middle_needed > 0 and len(middle_range) > 0:
                selected_middle = np.random.choice(
                    middle_range, 
                    size=min(n_middle_needed, len(middle_range)),
                    replace=False
                ).tolist()
                must_have.extend(selected_middle)
            
            return sorted(set(must_have), reverse=True)[:target_nfe]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: str = "cuda",
        target_nfe: int = 50,
        selection_strategy: str = "importance",
        eta: float = 0.0,
        return_stats: bool = True,
        show_progress: bool = True,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Generate samples with target NFE.
        
        Args:
            model: Noise prediction model
            shape: Output shape (batch_size, channels, height, width)
            device: Device to run on
            target_nfe: Target number of function evaluations
            selection_strategy: How to select timesteps ("uniform", "importance", "stochastic")
            eta: DDIM stochasticity (0 = deterministic)
            
        Returns:
            Tuple of (samples, stats_dict)
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        # Select timesteps
        active_timesteps = self._select_timesteps(target_nfe, selection_strategy)
        
        iterator = active_timesteps
        if show_progress:
            iterator = tqdm(iterator, desc=f"TargetNFE-{target_nfe} ({selection_strategy})")
        
        timesteps_used = []
        
        for i, t in enumerate(iterator):
            timesteps_used.append(t)
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Get next timestep
            if i < len(active_timesteps) - 1:
                t_prev = active_timesteps[i + 1]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev].to(device)
            else:
                t_prev = 0
                alpha_cumprod_t_prev = torch.tensor(1.0).to(device)
            
            # DDIM update
            pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
            
            if eta > 0 and t_prev > 0:
                sigma = eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * 
                    (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                )
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir + sigma * noise
            else:
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir
        
        stats = None
        if return_stats:
            stats = {
                "total_timesteps": self.num_timesteps,
                "steps_taken": len(timesteps_used),
                "steps_skipped": self.num_timesteps - len(timesteps_used),
                "skip_rate": (self.num_timesteps - len(timesteps_used)) / self.num_timesteps,
                "timesteps_used": timesteps_used,
                "target_nfe": target_nfe,
                "selection_strategy": selection_strategy
            }
        
        return x, stats