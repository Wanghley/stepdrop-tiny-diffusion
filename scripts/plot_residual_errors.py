#!/usr/bin/env python3
"""
Visualize the residual error between DDIM and StepDrop samples.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.modules import TinyUNet
from src.sampler import DDIMSampler, TargetNFEStepDropSampler

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

def plot_residuals():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    
    # Samplers
    ddim = DDIMSampler(num_timesteps=1000, num_inference_steps=50)
    stepdrop = TargetNFEStepDropSampler(num_timesteps=1000)
    
    # Fixed seed for same noise
    seed = 42
    shape = (1, 3, 32, 32)
    
    # Generate DDIM Baseline
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    with torch.no_grad():
        x_ddim = ddim.sample(model, shape, device=device, show_progress=False)
    
    # Generate StepDrop
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    with torch.no_grad():
        x_stepdrop, _ = stepdrop.sample(model, shape, device=device, target_nfe=50, selection_strategy="importance", show_progress=False)
    
    # Process images for display
    # 1. Clamp to [-1, 1]
    x_ddim = x_ddim.clamp(-1, 1).cpu()
    x_stepdrop = x_stepdrop.clamp(-1, 1).cpu()
    
    # 2. Compute absolute difference in normalized space [-1, 1]
    # We want visualizing the MAGNITUDE of error
    diff = (x_ddim - x_stepdrop).abs()
    
    # 3. For heatmap, we can average across channels or show RGB difference
    # Let's show average error intensity
    diff_map = diff.mean(dim=1).squeeze() # (32, 32)
    
    # 4. Convert images to [0, 1] for plotting
    img_ddim = (x_ddim.squeeze().permute(1, 2, 0) + 1) / 2
    img_stepdrop = (x_stepdrop.squeeze().permute(1, 2, 0) + 1) / 2
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # DDIM
    axes[0].imshow(img_ddim)
    axes[0].set_title('DDIM (50 steps)\nBaseline', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # StepDrop
    axes[1].imshow(img_stepdrop)
    axes[1].set_title('StepDrop (50 steps)\nImportance', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Residuals
    im = axes[2].imshow(diff_map, cmap='magma', vmin=0, vmax=0.5) # Cap vmax to make errors visible
    axes[2].set_title('Residual Error\n|DDIM - StepDrop|', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar
    dt = 0.5
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('results/plot_residuals.png', dpi=300, bbox_inches='tight')
    print("✅ Saved residual heatmap to results/plot_residuals.png")

if __name__ == "__main__":
    plot_residuals()
