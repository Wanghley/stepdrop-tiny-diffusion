"""
DDPM Sampler
============

Denoising Diffusion Probabilistic Models (DDPM) sampler.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
https://arxiv.org/abs/2006.11239
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm


class DDPMSampler:
    """
    Denoising Diffusion Probabilistic Models (DDPM) Sampler.
    
    Standard DDPM sampling requires all T timesteps, making it slow but 
    producing high-quality samples.
    
    Args:
        num_timesteps: Total number of diffusion timesteps (default: 1000)
        beta_schedule: Type of beta schedule ("cosine" or "linear")
        beta_start: Starting beta value for linear schedule
        beta_end: Ending beta value for linear schedule
    
    Example:
        >>> sampler = DDPMSampler(num_timesteps=1000)
        >>> samples = sampler.sample(model, (16, 3, 32, 32), device="cuda")
    """
    
    def __init__(
        self, 
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        
        # Compute beta schedule
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), 
            self.alphas_cumprod[:-1]
        ])
        
        # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        # Posterior mean coefficients
        # μ̃_t(x_t, x_0) = (√ᾱ_{t-1} * β_t) / (1 - ᾱ_t) * x_0 
        #                + (√α_t * (1 - ᾱ_{t-1})) / (1 - ᾱ_t) * x_t
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule from "Improved Denoising Diffusion Probabilistic Models".
        https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def _predict_x0_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: int, 
        noise: torch.Tensor,
        device: str
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        alpha_cumprod = self.alphas_cumprod[t].to(device)
        return (x_t - torch.sqrt(1 - alpha_cumprod) * noise) / torch.sqrt(alpha_cumprod)
    
    def _q_posterior_mean(
        self, 
        x_0: torch.Tensor, 
        x_t: torch.Tensor, 
        t: int,
        device: str
    ) -> torch.Tensor:
        """Compute posterior mean μ̃_t(x_t, x_0)."""
        coef1 = self.posterior_mean_coef1[t].to(device)
        coef2 = self.posterior_mean_coef2[t].to(device)
        return coef1 * x_0 + coef2 * x_t
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: str = "cuda",
        return_all_timesteps: bool = False,
        show_progress: bool = True,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Generate samples using DDPM reverse process.
        
        Args:
            model: Noise prediction model ε_θ(x_t, t)
            shape: Output shape (batch_size, channels, height, width)
            device: Device to run sampling on
            return_all_timesteps: If True, return all intermediate samples
            show_progress: If True, show progress bar
            clip_denoised: If True, clip predicted x_0 to [-1, 1]
        
        Returns:
            Generated samples of shape `shape`, or all timesteps if requested
        """
        batch_size = shape[0]
        
        # Start from pure noise: x_T ~ N(0, I)
        x = torch.randn(shape, device=device)
        
        all_samples = [x] if return_all_timesteps else None
        
        # Reverse process: t = T-1, T-2, ..., 0
        timesteps = reversed(range(self.num_timesteps))
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling", total=self.num_timesteps)
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise: ε_θ(x_t, t)
            predicted_noise = model(x, t_batch)
            
            # Predict x_0 from x_t and predicted noise
            pred_x0 = self._predict_x0_from_noise(x, t, predicted_noise, device)
            
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Compute posterior mean
            mean = self._q_posterior_mean(pred_x0, x, t, device)
            
            # Sample x_{t-1}
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[t].to(device)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
            
            if return_all_timesteps:
                all_samples.append(x)
        
        if return_all_timesteps:
            return torch.stack(all_samples)
        return x
    
    def get_nfe(self) -> int:
        """Return number of function evaluations (NFE) per sample."""
        return self.num_timesteps


def main():
    parser = argparse.ArgumentParser(description='DDPM Sampling')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--output_dir', type=str, default='./samples', help='Directory to save samples')
    parser.add_argument('--return_all', action='store_true', help='Return all timesteps')
    
    args = parser.parse_args()
    
    # Hard-coded parameters
    model_path = 'checkpoints/model.pt'
    image_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Initialize sampler
    print(f"Initializing DDPM sampler with {args.num_timesteps} timesteps")
    sampler = DDPMSampler(num_timesteps=args.num_timesteps)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    shape = (args.num_samples, args.channels, image_size, image_size)
    samples = sampler.sample(
        model=model,
        shape=shape,
        device=device,
        return_all_timesteps=args.return_all
    )
    
    # Save samples
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'ddpm_samples_t{args.num_timesteps}.pt'
    torch.save(samples, output_path)
    print(f"Samples saved to {output_path}")
    print(f"Sample shape: {samples.shape}")


if __name__ == "__main__":
    main()