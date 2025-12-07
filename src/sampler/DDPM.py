import torch
import torch.nn as nn
import argparse
from pathlib import Path

class DDPMSampler:  
    """Denoising Diffusion Probabilistic Models Sampling"""
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        # Returns a 1-D tensor of betas for each timestep
        self.betas = self._cosine_beta_schedule(num_timesteps)
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        # Computes cumulative product of alpha values across the rows
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Creates alpha values shifted by one timestep with initial value 1.0
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
    def sample(
        self, 
        model: nn.Module, 
        shape: tuple, 
        device: str = "cuda", 
        return_all_timesteps: bool = False
    ):
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