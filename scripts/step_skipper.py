import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, List


class StochasticStepSkipScheduler:
    
    def __init__(
        self,
        num_timesteps: int = 1000
    ):
        self.num_timesteps = num_timesteps

        self.betas = self._cosine_beta_schedule(num_timesteps)
        
        # Pre-compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _compute_skip_probability(self, t: int, strategy: str = "linear", base_prob: float = 0.3,**kwargs) -> float:
        progress = t / self.num_timesteps
        
        if strategy == "linear":
            # Higher skip probability at early timesteps (high noise)
            # Lower skip probability at late timesteps (fine details)
            return base_prob * progress
        
        elif strategy == "inverse_linear":
            # Lower skip probability at early timesteps
            # Higher skip probability at late timesteps
            return base_prob * (1 - progress)
        
        elif strategy == "exponential":
            # Exponentially increasing skip probability
            decay = kwargs.get("decay", 2.0)
            return base_prob * (progress ** decay)
        
        elif strategy == "sigmoid":
            # Sigmoid-shaped skip probability
            midpoint = kwargs.get("midpoint", 0.5)
            steepness = kwargs.get("steepness", 10.0)
            x = (progress - midpoint) * steepness
            return base_prob * (1 / (1 + np.exp(-x)))
        
        elif strategy == "adaptive":
            # Skip more in middle timesteps, less at extremes
            return base_prob * (1 - abs(2 * progress - 1))
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _compute_noise_magnitude(self, predicted_noise: torch.Tensor) -> float:
        """Compute magnitude of predicted noise for adaptive skipping."""
        return predicted_noise.abs().mean().item()
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        skip_strategy: str = "linear",
        base_skip_prob: float = 0.3,
        use_noise_adaptive: bool = False,
        noise_threshold: float = 0.1,
        device: str = "cuda",
        return_stats: bool = False,
        seed: Optional[int] = None,
        **strategy_kwargs
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Statistics tracking
        steps_executed = 0
        steps_skipped = 0
        skip_decisions = []
        skip_probs = []
        
        # Iteratively denoise
        for t in reversed(range(self.num_timesteps)):
            # Compute skip probability
            skip_prob = self._compute_skip_probability(
                t, skip_strategy, base_skip_prob, **strategy_kwargs
            )
            skip_probs.append(skip_prob)
            
            # Decide whether to skip this step
            should_skip = np.random.rand() < skip_prob
            
            # Noise-adaptive adjustment
            if use_noise_adaptive and not should_skip:
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                predicted_noise = model(x, t_batch)
                noise_mag = self._compute_noise_magnitude(predicted_noise)
                
                # If noise is very small, increase chance of skipping
                if noise_mag < noise_threshold:
                    should_skip = np.random.rand() < 0.8  # High prob of skipping low-noise steps
            
            skip_decisions.append(should_skip)
            
            if should_skip:
                steps_skipped += 1
                continue
            
            steps_executed += 1
            
            # Execute denoising step
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            
            # Get alpha values
            alpha = self.alphas[t].to(device)
            alpha_cumprod = self.alphas_cumprod[t].to(device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Compute posterior mean and variance
            if t > 0:
                alpha_cumprod_prev = self.alphas_cumprod_prev[t].to(device)
                posterior_variance = (
                    self.betas[t] * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
                ).to(device)
                posterior_mean_coef1 = (
                    self.betas[t] * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
                ).to(device)
                posterior_mean_coef2 = (
                    (1.0 - alpha_cumprod_prev) * torch.sqrt(alpha) / (1.0 - alpha_cumprod)
                ).to(device)
                
                mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_variance) * noise
            else:
                # Final step: no noise
                x = pred_x0
        
        if return_stats:
            stats = {
                "steps_executed": steps_executed,
                "steps_skipped": steps_skipped,
                "total_steps": self.num_timesteps,
                "skip_rate": steps_skipped / self.num_timesteps,
                "speedup": self.num_timesteps / steps_executed if steps_executed > 0 else 0,
                "skip_decisions": skip_decisions,
                "skip_probabilities": skip_probs
            }
            return x, stats
        
        return x


class AdaptiveStepSkipScheduler(StochasticStepSkipScheduler):
    """
    Advanced version that adapts skip probability based on prediction quality.
    
    Uses a running estimate of denoising effectiveness to decide which steps to skip.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_quality_history = []
    
    def _estimate_prediction_quality(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: int,
        device: str
    ) -> float:
        """
        Estimate quality of the noise prediction.
        Higher quality = more confident we can skip nearby steps.
        """
        alpha_cumprod = self.alphas_cumprod[t].to(device)
        
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
        
        # Measure smoothness (good predictions should be smooth)
        if pred_x0.dim() == 4:  # Image data
            # Compute gradient magnitude
            dx = torch.abs(pred_x0[:, :, :, 1:] - pred_x0[:, :, :, :-1])
            dy = torch.abs(pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :])
            smoothness = -(dx.mean() + dy.mean()).item()  # Negative because higher smoothness = better
        else:
            # For non-image data, use variance as a proxy
            smoothness = -pred_x0.var().item()
        
        return smoothness
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        base_skip_prob: float = 0.2,
        quality_threshold: float = -0.1,
        adaptation_rate: float = 0.1,
        device: str = "cuda",
        return_stats: bool = False,
        seed: Optional[int] = None
    ):
        """
        Sample with quality-adaptive step skipping.
        
        Args:
            model: Denoising model
            shape: Shape of samples
            base_skip_prob: Base skip probability
            quality_threshold: Quality threshold for enabling skips
            adaptation_rate: How quickly to adapt skip probability
            device: Device to run on
            return_stats: Return sampling statistics
            seed: Random seed
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        steps_executed = 0
        steps_skipped = 0
        quality_scores = []
        current_skip_prob = base_skip_prob
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            
            # Estimate prediction quality
            quality = self._estimate_prediction_quality(x, predicted_noise, t, device)
            quality_scores.append(quality)
            
            # Adapt skip probability based on quality
            if quality > quality_threshold:
                current_skip_prob = min(0.8, current_skip_prob + adaptation_rate)
            else:
                current_skip_prob = max(0.0, current_skip_prob - adaptation_rate)
            
            # Decide whether to skip
            should_skip = np.random.rand() < current_skip_prob and t > 10  # Never skip last 10 steps
            
            if should_skip:
                steps_skipped += 1
                continue
            
            steps_executed += 1
            
            # Execute denoising step (same as before)
            alpha = self.alphas[t].to(device)
            alpha_cumprod = self.alphas_cumprod[t].to(device)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            if t > 0:
                alpha_cumprod_prev = self.alphas_cumprod_prev[t].to(device)
                posterior_variance = (
                    self.betas[t] * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
                ).to(device)
                posterior_mean_coef1 = (
                    self.betas[t] * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
                ).to(device)
                posterior_mean_coef2 = (
                    (1.0 - alpha_cumprod_prev) * torch.sqrt(alpha) / (1.0 - alpha_cumprod)
                ).to(device)
                
                mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_variance) * noise
            else:
                x = pred_x0
        
        if return_stats:
            stats = {
                "steps_executed": steps_executed,
                "steps_skipped": steps_skipped,
                "skip_rate": steps_skipped / self.num_timesteps,
                "speedup": self.num_timesteps / steps_executed if steps_executed > 0 else 0,
                "quality_scores": quality_scores,
                "avg_quality": np.mean(quality_scores)
            }
            return x, stats
        
        return x


# Example usage and comparison
if __name__ == "__main__":
    # Mock model
    class SimpleNoisePredictor(nn.Module):
        def forward(self, x, t):
            return torch.randn_like(x) * 0.1
    
    model = SimpleNoisePredictor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print("=" * 60)
    print("STOCHASTIC STEP-SKIP SAMPLING COMPARISON")
    print("=" * 60)
    
    # Test different skip strategies
    strategies = ["uniform", "linear", "inverse_linear", "adaptive"]
    
    scheduler = StochasticStepSkipScheduler(num_timesteps=1000)
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        samples, stats = scheduler.sample(
            model=model,
            shape=(4, 3, 32, 32),
            skip_strategy=strategy,
            base_skip_prob=0.3,
            device=device,
            return_stats=True,
            seed=42
        )
        
        print(f"  Steps executed: {stats['steps_executed']}")
        print(f"  Steps skipped: {stats['steps_skipped']}")
        print(f"  Skip rate: {stats['skip_rate']:.2%}")
        print(f"  Speedup: {stats['speedup']:.2f}x")
    
    # Test adaptive quality-based skipping
    print("\n" + "=" * 60)
    print("ADAPTIVE QUALITY-BASED SKIPPING")
    print("=" * 60)
    
    adaptive_scheduler = AdaptiveStepSkipScheduler(num_timesteps=1000)
    samples, stats = adaptive_scheduler.sample(
        model=model,
        shape=(4, 3, 32, 32),
        base_skip_prob=0.2,
        device=device,
        return_stats=True,
        seed=42
    )
    
    print(f"Steps executed: {stats['steps_executed']}")
    print(f"Steps skipped: {stats['steps_skipped']}")
    print(f"Skip rate: {stats['skip_rate']:.2%}")
    print(f"Speedup: {stats['speedup']:.2f}x")
    print(f"Average quality: {stats['avg_quality']:.4f}")