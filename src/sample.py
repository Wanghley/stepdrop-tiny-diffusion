import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import conf
from modules import TinyUNet
from scheduler import NoiseScheduler

@torch.no_grad()
def sample_ddpm(model, scheduler, n_samples=16, show_progress=True):
    """Standard DDPM sampling: T steps"""
    print(f"Sampling DDPM ({scheduler.n_timesteps} steps)...")
    model.eval()
    
    # Start from pure noise
    x = torch.randn((n_samples, conf.channels, conf.img_size, conf.img_size)).to(conf.device)
    
    timesteps = range(scheduler.n_timesteps - 1, -1, -1)
    if show_progress:
        timesteps = tqdm(timesteps, desc="DDPM Sampling")
    
    for t in timesteps:
        t_batch = torch.full((n_samples,), t, device=conf.device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Get scheduler values
        alpha = scheduler.alphas[t]
        alpha_hat = scheduler.alpha_hats[t]
        beta = scheduler.betas[t]
        
        # Compute mean
        # μ_θ(x_t, t) = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_hat)) * predicted_noise
        )
        
        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(scheduler.posterior_variance[t])
            x = mean + sigma * noise
        else:
            x = mean
    
    # Denormalize from [-1, 1] to [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2
    return x


@torch.no_grad()
def sample_ddim(model, scheduler, n_samples=16, steps=50, eta=0.0, show_progress=True):
    """DDIM sampling: Accelerated deterministic/stochastic sampling"""
    print(f"Sampling DDIM ({steps} steps, eta={eta})...")
    model.eval()
    
    # Start from pure noise
    x = torch.randn((n_samples, conf.channels, conf.img_size, conf.img_size)).to(conf.device)
    
    # Create subsequence of timesteps
    step_ratio = scheduler.n_timesteps // steps
    timesteps = list(range(0, scheduler.n_timesteps, step_ratio))[::-1]
    
    iterator = timesteps
    if show_progress:
        iterator = tqdm(timesteps, desc="DDIM Sampling")
    
    for i, t in enumerate(iterator):
        t_batch = torch.full((n_samples,), t, device=conf.device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Get alpha values
        alpha_hat_t = scheduler.alpha_hats[t]
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clip for stability
        
        # Get previous timestep
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_hat_t_prev = scheduler.alpha_hats[t_prev]
        else:
            alpha_hat_t_prev = torch.tensor(1.0).to(conf.device)
        
        # Compute variance
        sigma_t = eta * torch.sqrt(
            (1 - alpha_hat_t_prev) / (1 - alpha_hat_t) *
            (1 - alpha_hat_t / alpha_hat_t_prev)
        )
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_hat_t_prev - sigma_t**2) * predicted_noise
        
        # Add noise if eta > 0 and not final step
        if i < len(timesteps) - 1 and eta > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        # Compute x_{t-1}
        x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt + sigma_t * noise
    
    # Denormalize from [-1, 1] to [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2
    return x


def show_images(images, title="Generated Images", save_path="generated.png"):
    """Helper to visualize grid of images"""
    n = min(len(images), 16)
    nrow = int(n ** 0.5)
    ncol = (n + nrow - 1) // nrow
    
    images = images[:n].cpu()
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2 * nrow))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i].permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Load Model
    print(f"Loading model from {conf.save_path}...")
    model = TinyUNet(
        img_size=conf.img_size, 
        channels=conf.channels, 
        base_channels=conf.base_channels
    ).to(conf.device)
    model.load_state_dict(torch.load(conf.save_path, map_location=conf.device))
    model.eval()
    
    scheduler = NoiseScheduler(
        n_timesteps=conf.n_timesteps, 
        schedule_type="cosine",
        device=conf.device
    )
    
    print(f"Device: {conf.device}")
    
    # 1. Sample with DDPM (high quality, slow)
    print("\n" + "="*50)
    print("DDPM Sampling (1000 steps)")
    print("="*50)
    images_ddpm = sample_ddpm(model, scheduler, n_samples=16)
    show_images(images_ddpm, "DDPM Results (1000 steps)", "ddpm_samples.png")
    
    # 2. Sample with DDIM (fast)
    print("\n" + "="*50)
    print("DDIM Sampling (50 steps)")
    print("="*50)
    images_ddim = sample_ddim(model, scheduler, n_samples=16, steps=50, eta=0.0)
    show_images(images_ddim, "DDIM Results (50 steps)", "ddim_samples.png")
    
    # 3. Sample with DDIM (very fast)
    print("\n" + "="*50)
    print("DDIM Sampling (20 steps)")
    print("="*50)
    images_ddim_fast = sample_ddim(model, scheduler, n_samples=16, steps=20, eta=0.0)
    show_images(images_ddim_fast, "DDIM Results (20 steps)", "ddim_fast_samples.png")