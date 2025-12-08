#!/usr/bin/env python3
"""
Visualize the "Barcode" of selected timesteps for different strategies.
"""

import matplotlib.pyplot as plt
import numpy as np
from src.sampler import TargetNFEStepDropSampler

# Duke Colors
DUKE_BLUE = '#012169'
DUKE_ROYAL_BLUE = '#00539B'
DUKE_LIGHT_BLUE = '#668CFF' # Lighter for better contrast in barcode

def plot_timestep_barcode():
    sampler = TargetNFEStepDropSampler(num_timesteps=1000)
    target_nfe = 50
    
    strategies = ["uniform", "importance", "stochastic"]
    labels = ["Uniform (Baseline)", "Importance (Ours)", "Stochastic"]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for i, strategy in enumerate(strategies):
        # Get selected timesteps
        if strategy == "stochastic":
            # Averaging stochastic runs isn't great for a static plot, 
            # so we show one representative run
            np.random.seed(42) 
        
        timesteps = sampler._select_timesteps(target_nfe, strategy=strategy)
        
        # Plot as vertical lines (barcode)
        y_pos = len(strategies) - 1 - i
        
        # Draw base line
        ax.hlines(y_pos, 0, 1000, color='gray', alpha=0.2, linewidth=1)
        
        # Draw steps
        ax.vlines(timesteps, y_pos - 0.3, y_pos + 0.3, colors=DUKE_BLUE, linewidth=1.5, alpha=0.8)
        
        # Label
        ax.text(-50, y_pos, labels[i], ha='right', va='center', fontsize=12, fontweight='bold', color='black')
        
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel('Diffusion Timestep t (0=Clean, 1000=Noisy)', fontsize=12, fontweight='bold')
    ax.set_xlim(-150, 1050)
    ax.set_title(f'Timestep Selection Strategies (NFE={target_nfe})', fontsize=14, fontweight='bold', color=DUKE_BLUE)
    
    # Add annotations
    ax.annotate('Focus on\nFine Details', (50, 2.5), xytext=(50, 3.2), 
                ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.annotate('Focus on\nStructure', (950, 2.5), xytext=(950, 3.2), 
                ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
                
    plt.tight_layout()
    plt.savefig('results/plot_timestep_barcode.png', dpi=300, bbox_inches='tight')
    print("✅ Saved barcode plot to results/plot_timestep_barcode.png")

if __name__ == "__main__":
    plot_timestep_barcode()
