import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable

class DDPMSampler:
    
    def __init__(
        self,
        num_timesteps: int = 1000,
    ):
        self.num_timesteps = num_timesteps

        self.betas = self._cosine_beta_schedule(num_timesteps)

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple, device: str = "cuda", return_all_timesteps: bool = False):
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        all_samples = [x] if return_all_timesteps else None
        
        # Iteratively denoise
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Compute denoised sample
            alpha = self.alphas[t].to(device)
            alpha_cumprod = self.alphas_cumprod[t].to(device)
            beta = self.betas[t].to(device)
            
            # Mean of p(x_{t-1} | x_t)
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            pred_x0 = torch.clamp(pred_x0, -1, 1)  # Optional: clip for stability
            
            # Compute mean using posterior formula
            posterior_mean_coef1 = self.posterior_mean_coef1[t].to(device)
            posterior_mean_coef2 = self.posterior_mean_coef2[t].to(device)
            posterior_variance = self.posterior_variance[t].to(device)
            
            mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x
            
            # Sample x_{t-1}
            x = mean + torch.sqrt(posterior_variance) * noise
            
            if return_all_timesteps:
                all_samples.append(x)
        
        if return_all_timesteps:
            return torch.stack(all_samples)
        return x


class DDIMSampler:
    
    def __init__(
        self,
        num_timesteps: int = 1000,
    ):
        self.num_timesteps = num_timesteps
    
        self.betas = self._cosine_beta_schedule(num_timesteps)
        
        # Pre-compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple, num_inference_steps: int = 50, eta: 
               float = 0.0, device: str = "cuda", return_all_timesteps: bool = False):
        
        batch_size = shape[0]
        
        # Create subsequence of timesteps
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps = torch.from_numpy(timesteps).to(device)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        all_samples = [x] if return_all_timesteps else None
        
        # Iteratively denoise
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_batch)
            
            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)  # Optional: clip for stability
            
            # Get previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev].to(device)
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0).to(device)
            
            # Compute variance
            sigma_t = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            
            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise
            
            # Add noise
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


if __name__ == "__main__":
    class SimpleNoisePredictor(nn.Module):
        def forward(self, x, t):
            # In practice, this would be a U-Net or similar architecture
            return torch.randn_like(x)
    
    model = SimpleNoisePredictor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print("Running DDPM sampling...")
    ddpm_sampler = DDPMSampler(num_timesteps=1000)
    samples_ddpm = ddpm_sampler.sample(
        model=model,
        shape=(4, 3, 32, 32),  # 4 images, 3 channels, 32x32
        device=device
    )
    
    print("\nRunning DDIM sampling (deterministic)...")
    ddim_sampler = DDIMSampler(num_timesteps=1000)
    samples_ddim = ddim_sampler.sample(
        model=model,
        shape=(4, 3, 32, 32),
        num_inference_steps=50,
        eta=0.0,  # Deterministic
        device=device
    )
    
    # DDIM sampling (50 steps, stochastic)
    print("\nRunning DDIM sampling (stochastic)...")
    samples_ddim_stochastic = ddim_sampler.sample(
        model=model,
        shape=(4, 3, 32, 32),
        num_inference_steps=50,
        eta=1.0,  # Stochastic (similar to DDPM)
        device=device
    )