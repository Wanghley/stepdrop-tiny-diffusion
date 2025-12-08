#!/usr/bin/env python3
"""
Generate a comparison grid of samples from different strategies using the SAME random seed.
"""

import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
from src.modules import TinyUNet
from src.sampler import DDPMSampler, DDIMSampler, TargetNFEStepDropSampler

def get_sampler(strategy_name, num_timesteps=1000):
    if strategy_name == "DDPM_1000":
        return DDPMSampler(num_timesteps=1000)
    elif strategy_name == "DDIM_50":
        return DDIMSampler(num_timesteps=1000, num_inference_steps=50)
    elif strategy_name == "StepDrop_Target50":
        return TargetNFEStepDropSampler(num_timesteps=1000)
    elif strategy_name == "StepDrop_Target25":
        return TargetNFEStepDropSampler(num_timesteps=1000)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def run_sampler(sampler, model, strategy_name, device):
    shape = (1, 3, 32, 32)
    
    if strategy_name == "DDPM_1000":
        return sampler.sample(model, shape, device=device, show_progress=False)
    elif strategy_name == "DDIM_50":
        return sampler.sample(model, shape, device=device, show_progress=False)
    elif strategy_name == "StepDrop_Target50":
        return sampler.sample(model, shape, device=device, target_nfe=50, selection_strategy="importance", show_progress=False)[0]
    elif strategy_name == "StepDrop_Target25":
        return sampler.sample(model, shape, device=device, target_nfe=25, selection_strategy="importance", show_progress=False)[0]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "cifar10_64ch_50ep.pt"
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # Fix warning here too
    
    config = checkpoint.get('config', {'img_size': 32, 'channels': 3, 'base_channels': 64})
    model = TinyUNet(img_size=config['img_size'], channels=config['channels'], base_channels=config['base_channels'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    strategies = ["DDPM_1000", "DDIM_50", "StepDrop_Target50", "StepDrop_Target25"]
    labels = ["DDPM (1000)", "DDIM (50)", "StepDrop (50)", "StepDrop (25)"]
    
    num_rows = 4
    grid_images = []
    
    print("Generating comparison grid...")
    
    # We want rows to be seeds, cols to be strategies
    # To construct the grid easily with make_grid, we generally want a flat list: 
    # [Seed1_Strat1, Seed1_Strat2, ..., Seed2_Strat1, ...]
    
    for row_idx in range(num_rows):
        seed = 42 + row_idx # Fixed seeds for reproducibility
        print(f"Generating row {row_idx+1}/{num_rows} (Seed {seed})")
        
        for strategy in strategies:
            # Set seed IMMEDIATELY before generation to ensure same initial noise
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                
            sampler = get_sampler(strategy)
            
            with torch.no_grad():
                sample = run_sampler(sampler, model, strategy, device)
                
            # Clamp and normalize to [0, 1]
            sample = (sample.clamp(-1, 1) + 1) / 2
            grid_images.append(sample.cpu())

    # Concatenate all images
    all_images = torch.cat(grid_images, dim=0)
    
    # Create grid
    grid = torchvision.utils.make_grid(all_images, nrow=len(strategies), padding=2, pad_value=1.0)
    
    # Plot using matplotlib to add labels
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    
    # Add column headers
    grid_w = grid.shape[2]
    col_w = grid_w / len(strategies)
    for i, label in enumerate(labels):
        plt.text((i + 0.5) * col_w, -10, label, ha='center', fontsize=12, fontweight='bold')
        
    output_path = "results/comparison_grid.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved comparison grid to {output_path}")

if __name__ == "__main__":
    main()
