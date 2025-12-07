import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path

class DDIMSampler:
    """Denoising Diffusion Implicit Models Sampling"""
    def __init__(self, num_timesteps: int = 1000, num_inference_steps: int = 50, eta: float = 0.0):
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
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
    def sample(
        self, 
        model: nn.Module, 
        shape: tuple, 
        num_inference_steps: int = None, 
        eta: float = None, 
        device: str = "cuda", 
        return_all_timesteps: bool = False
    ):
        batch_size = shape[0]
        
        # Use instance defaults if not provided
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if eta is None:
            eta = self.eta
        
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