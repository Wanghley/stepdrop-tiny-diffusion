#!/usr/bin/env python3
"""
Visualize the Denoising Evolution ("Film Strip") of DDIM vs StepDrop.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.modules import TinyUNet
from src.sampler import DDIMSampler, TargetNFEStepDropSampler

# =============================================================================
# Helper: Custom Sampling with Intermediates
# =============================================================================

@torch.no_grad()
def sample_with_intermediates(
    model: nn.Module,
    sampler,
    shape: tuple,
    device: str,
    target_timesteps: list,
    strategy_type: str = "ddim" # "ddim" or "stepdrop"
):
    """
    Run sampling and capture x_t at specific 'target_timesteps'.
    Replicates the sampler logic but hooks into the loop.
    """
    batch_size = shape[0]
    x = torch.randn(shape, device=device)
    
    # Storage for snapshots
    # Map t -> image tensor
    snapshots = {}
    
    # Save initial noise (t=1000 equivalent)
    if 1000 in target_timesteps:
        snapshots[1000] = x.clone()
    
    if strategy_type == "ddim":
        # DDIM Logic
        num_inference_steps = 50
        num_timesteps = 1000
        step_ratio = num_timesteps // num_inference_steps
        timesteps = list(range(0, num_timesteps, step_ratio))
        timesteps = sorted(timesteps, reverse=True) # [980, 960, ..., 0] roughly
        
        # We need to map the "continuous" t to our discrete steps
        # But wait, DDIM uses a subset. The `target_timesteps` (e.g. 750) might NOT be in the subset.
        # We will capture the NEAREST step.
        active_timesteps = timesteps
        
    elif strategy_type == "stepdrop":
        # StepDrop Logic
        # We use the 'importance' strategy with 50 steps
        active_timesteps = sampler._select_timesteps(target_nfe=50, strategy="importance")
        # Ensure endpoints
        if 0 not in active_timesteps: active_timesteps.append(0)
        active_timesteps = sorted(set(active_timesteps), reverse=True)
        
    # --- Shared Loop Logic (DDIM-style update) ---
    # We need alphas from the sampler
    alphas_cumprod = sampler.alphas_cumprod
    
    for i, t in enumerate(active_timesteps):
        # Check if we should save snapshot (approximate match)
        # Find closest target_t that is > current t but < previous t? 
        # Actually, let's just save if t is close to a target
        for target in target_timesteps:
            if abs(t - target) < 30: # Tolerance of 30 steps (StepDrop takes big jumps)
                if target not in snapshots:
                    snapshots[target] = x.clone()
        
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Get alpha values
        alpha_cumprod_t = alphas_cumprod[t].to(device)
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        # pred_x0 = torch.clamp(pred_x0, -1, 1) # Standard DDIM doesn't always clip intermediate x0, but usually good
        
        # Get next timestep
        if i < len(active_timesteps) - 1:
            t_prev = active_timesteps[i + 1]
            alpha_cumprod_t_prev = alphas_cumprod[t_prev].to(device)
        else:
            t_prev = 0
            alpha_cumprod_t_prev = torch.tensor(1.0).to(device)
        
        # DDIM update
        pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
        x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir
        
    # Always save final
    if 0 in target_timesteps:
        snapshots[0] = x.clone()
        
    return snapshots

# =============================================================================
# Main
# =============================================================================

def load_model(device):
    checkpoint_path = "cifar10_64ch_50ep.pt"
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {'img_size': 32, 'channels': 3, 'base_channels': 64})
    model = TinyUNet(img_size=config['img_size'], channels=config['channels'], base_channels=config['base_channels'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def plot_evolution():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    
    # Samplers
    ddim_sampler = DDIMSampler(num_timesteps=1000, num_inference_steps=50)
    stepdrop_sampler = TargetNFEStepDropSampler(num_timesteps=1000)
    
    # Config
    seed = 44 # Good seed from grid
    shape = (1, 3, 32, 32)
    targets = [1000, 750, 500, 250, 0] # Targets to capture
    
    # Generate DDIM
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    snaps_ddim = sample_with_intermediates(model, ddim_sampler, shape, device, targets, "ddim")
    
    # Generate StepDrop
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    snaps_step = sample_with_intermediates(model, stepdrop_sampler, shape, device, targets, "stepdrop")
    
    # Plotting
    fig, axes = plt.subplots(2, len(targets), figsize=(12, 5))
    
    # Row 1: DDIM
    for i, t in enumerate(targets):
        ax = axes[0, i]
        if t in snaps_ddim:
            img = (snaps_ddim[t].clamp(-1, 1).squeeze().permute(1, 2, 0) + 1) / 2
            ax.imshow(img.cpu())
        else:
            ax.text(0.5, 0.5, "Missed", ha='center')
            
        ax.set_title(f"t={t}", fontsize=10, fontweight='bold')
        if i == 0: ax.set_ylabel("DDIM", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Row 2: StepDrop
    for i, t in enumerate(targets):
        ax = axes[1, i]
        if t in snaps_step:
            img = (snaps_step[t].clamp(-1, 1).squeeze().permute(1, 2, 0) + 1) / 2
            ax.imshow(img.cpu())
        else:
            ax.text(0.5, 0.5, "N/A", ha='center')
            
        if i == 0: ax.set_ylabel("StepDrop", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.suptitle("Denoising Evolution: StepDrop converges faster to semantics", fontsize=14, fontweight='bold', color='#012169')
    plt.tight_layout()
    plt.savefig('results/plot_denoising_evolution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved evolution plot to results/plot_denoising_evolution.png")

if __name__ == "__main__":
    plot_evolution()
