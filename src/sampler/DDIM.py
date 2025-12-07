"""
DDIM Sampler
============

Denoising Diffusion Implicit Models (DDIM) sampler.

Reference: Song et al., "Denoising Diffusion Implicit Models" (2020)
https://arxiv.org/abs/2010.02502
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from tqdm import tqdm


class DDIMSampler:
    """
    Denoising Diffusion Implicit Models (DDIM) Sampler.
    
    DDIM enables faster sampling by skipping timesteps while maintaining
    sample quality. With eta=0, the process is deterministic.
    
    Args:
        num_timesteps: Total number of diffusion timesteps (default: 1000)
        num_inference_steps: Number of steps for sampling (default: 50)
        eta: Stochasticity parameter (0=deterministic, 1=DDPM-like)
        beta_schedule: Type of beta schedule ("cosine" or "linear")
    
    Example:
        >>> sampler = DDIMSampler(num_timesteps=1000, num_inference_steps=50)
        >>> samples = sampler.sample(model, (16, 3, 32, 32), device="cuda")
    """
    
    def __init__(
        self, 
        num_timesteps: int = 1000, 
        num_inference_steps: int = 50, 
        eta: float = 0.0,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
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
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule from Improved DDPM paper."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def _get_timesteps(
        self, 
        num_inference_steps: int,
        schedule: str = "uniform"
    ) -> np.ndarray:
        """
        Get timestep schedule for sampling.
        
        Args:
            num_inference_steps: Number of sampling steps
            schedule: Schedule type ("uniform", "quadratic", "cosine")
        
        Returns:
            Array of timesteps from high to low
        """
        if schedule == "uniform":
            step_ratio = self.num_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1]
        elif schedule == "quadratic":
            t_normalized = np.linspace(0, 1, num_inference_steps)
            timesteps = (self.num_timesteps - 1) * (1 - t_normalized) ** 2
        elif schedule == "cosine":
            t_normalized = np.linspace(0, 1, num_inference_steps)
            timesteps = (self.num_timesteps - 1) * (np.cos(t_normalized * np.pi / 2) ** 2)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        return timesteps.round().astype(np.int64)
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        num_inference_steps: Optional[int] = None,
        eta: Optional[float] = None,
        device: str = "cuda",
        return_all_timesteps: bool = False,
        show_progress: bool = True,
        clip_denoised: bool = True,
        schedule: str = "uniform"
    ) -> torch.Tensor:
        """
        Generate samples using DDIM reverse process.
        
        Args:
            model: Noise prediction model ε_θ(x_t, t)
            shape: Output shape (batch_size, channels, height, width)
            num_inference_steps: Override default inference steps
            eta: Override default eta (0=deterministic, 1=stochastic)
            device: Device to run sampling on
            return_all_timesteps: If True, return all intermediate samples
            show_progress: If True, show progress bar
            clip_denoised: If True, clip predicted x_0 to [-1, 1]
            schedule: Timestep schedule ("uniform", "quadratic", "cosine")
        
        Returns:
            Generated samples of shape `shape`
        """
        batch_size = shape[0]
        
        # Use instance defaults if not provided
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if eta is None:
            eta = self.eta
        
        # Get timestep schedule
        timesteps = self._get_timesteps(num_inference_steps, schedule)
        timesteps = torch.from_numpy(timesteps).to(device)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        all_samples = [x] if return_all_timesteps else None
        
        # Reverse process
        iterator = enumerate(timesteps)
        if show_progress:
            iterator = tqdm(iterator, desc="DDIM Sampling", total=len(timesteps))
        
        for i, t in iterator:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)
            
            # Predict x_0: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Get previous timestep alpha
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev].to(device)
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)
            
            # Compute variance: σ_t = η * √((1-ᾱ_{t-1})/(1-ᾱ_t)) * √(1-ᾱ_t/ᾱ_{t-1})
            sigma_t = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise
            
            # Add noise (only if not last step and eta > 0)
            if i < len(timesteps) - 1 and eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # Compute x_{t-1}
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma_t * noise
            
            if return_all_timesteps:
                all_samples.append(x)
        
        if return_all_timesteps:
            return torch.stack(all_samples)
        return x
    
    def get_nfe(self) -> int:
        """Return number of function evaluations (NFE) per sample."""
        return self.num_inference_steps


def main():
    parser = argparse.ArgumentParser(description='DDIM Sampling')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Total number of diffusion timesteps')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of sampling steps (accelerated)')
    parser.add_argument('--eta', type=float, default=0.0, help='Stochasticity parameter (0=deterministic, 1=stochastic like DDPM)')
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
    print(f"Initializing DDIM sampler with {args.num_timesteps} total timesteps")
    print(f"Using {args.num_inference_steps} inference steps (eta={args.eta})")
    sampler = DDIMSampler(num_timesteps=args.num_timesteps)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    shape = (args.num_samples, args.channels, image_size, image_size)
    samples = sampler.sample(
        model=model,
        shape=shape,
        num_inference_steps=args.num_inference_steps,
        eta=args.eta,
        device=device,
        return_all_timesteps=args.return_all
    )
    
    # Save samples
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'ddim_samples_t{args.num_timesteps}_steps{args.num_inference_steps}_eta{args.eta}.pt'
    torch.save(samples, output_path)
    print(f"Samples saved to {output_path}")
    print(f"Sample shape: {samples.shape}")


if __name__ == "__main__":
    main()