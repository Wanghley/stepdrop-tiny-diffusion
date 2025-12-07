import torch

class NoiseScheduler:
    def __init__(self, n_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.n_timesteps = n_timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        
    def noise_images(self, x, t):
        """Forward Diffusion Process: Add noise to clean image"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hats[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hats[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """Randomly sample t for training"""
        return torch.randint(low=1, high=self.n_timesteps, size=(n,)).to(self.device)