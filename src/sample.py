import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import conf
from modules import TinyUNet
from scheduler import NoiseScheduler

def sample_ddpm(model, scheduler, n_samples=16):
    """Standard slow sampling: T steps"""
    print(f"Sampling DDPM ({conf.n_timesteps} steps)...")
    model.eval()
    with torch.no_grad():
        x = torch.randn((n_samples, conf.channels, conf.img_size, conf.img_size)).to(conf.device)
        
        for i in tqdm(reversed(range(1, conf.n_timesteps)), desc="DDPM"):
            t = (torch.ones(n_samples) * i).long().to(conf.device)
            predicted_noise = model(x, t)
            
            alpha = scheduler.alphas[t][:, None, None, None]
            alpha_hat = scheduler.alpha_hats[t][:, None, None, None]
            beta = scheduler.betas[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
    model.train()
    x = (x.clamp(-1, 1) + 1) / 2 # Normalize to [0, 1]
    return x

def sample_ddim(model, scheduler, n_samples=16, steps=50):
    """Deterministic fast sampling: Subset of steps"""
    print(f"Sampling DDIM ({steps} steps)...")
    model.eval()
    with torch.no_grad():
        x = torch.randn((n_samples, conf.channels, conf.img_size, conf.img_size)).to(conf.device)
        
        # Create a linear skipping schedule
        time_steps = torch.linspace(conf.n_timesteps - 1, 0, steps).long().to(conf.device)
        
        for i in tqdm(range(len(time_steps) - 1), desc="DDIM"):
            curr_t = time_steps[i]
            next_t = time_steps[i+1]
            
            t_tensor = (torch.ones(n_samples) * curr_t).long().to(conf.device)
            predicted_noise = model(x, t_tensor)
            
            alpha_hat_curr = scheduler.alpha_hats[curr_t]
            alpha_hat_next = scheduler.alpha_hats[next_t]
            
            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_hat_curr) * predicted_noise) / torch.sqrt(alpha_hat_curr)
            
            # Point to xt-1
            dir_xt = torch.sqrt(1 - alpha_hat_next) * predicted_noise
            
            # Update x
            x = torch.sqrt(alpha_hat_next) * pred_x0 + dir_xt
            
    model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    return x

def show_images(images, title="Generated Images"):
    """Helper to visualize grid"""
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    # Load Model
    model = TinyUNet(img_size=conf.img_size, channels=conf.channels, base_channels=conf.base_channels).to(conf.device)
    model.load_state_dict(torch.load(conf.save_path))
    scheduler = NoiseScheduler(n_timesteps=conf.n_timesteps, device=conf.device)
    
    # 1. Run DDPM (Baseline 1: High Quality, Slow)
    # images_ddpm = sample_ddpm(model, scheduler, n_samples=16)
    # show_images(images_ddpm, "DDPM Results")

    # 2. Run DDIM (Baseline 2: Low Latency, Fast)
    images_ddim = sample_ddim(model, scheduler, n_samples=16, steps=20)
    show_images(images_ddim, "DDIM Results (20 Steps)")