import torch
import torch.nn.functional as F


class NoiseScheduler:
    """Noise scheduler for diffusion models."""
    
    def __init__(
        self, 
        n_timesteps: int = 1000, 
        beta_start: float = 1e-4, 
        beta_end: float = 0.02, 
        schedule_type: str = "cosine", 
        device: str = "cpu"
    ):
        self.n_timesteps = n_timesteps
        self.device = device
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(n_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        self.alpha_hats_prev = F.pad(self.alpha_hats[:-1], (1, 0), value=1.0)
        
        # Pre-compute values
        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hats)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1.0 - self.alpha_hats)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance
        self.posterior_variance = self.betas * (1.0 - self.alpha_hats_prev) / (1.0 - self.alpha_hats)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)
        
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_hats_prev) / (1.0 - self.alpha_hats)
        self.posterior_mean_coef2 = (1.0 - self.alpha_hats_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_hats)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule from https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def noise_images(self, x: torch.Tensor, t: torch.Tensor) -> tuple:
        """Forward diffusion: q(x_t | x_0)"""
        sqrt_alpha_hat = self.sqrt_alpha_hats[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hats[t][:, None, None, None]
        epsilon = torch.randn_like(x)
        noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return noisy, epsilon
    
    def sample_timesteps(self, n: int) -> torch.Tensor:
        """Randomly sample timesteps for training."""
        return torch.randint(low=0, high=self.n_timesteps, size=(n,)).to(self.device)
    
    def get_index_from_list(self, vals, t, x_shape):
        """Helper to extract values at timestep t"""
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))