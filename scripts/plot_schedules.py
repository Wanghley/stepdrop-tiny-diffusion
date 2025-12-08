#!/usr/bin/env python3
"""
Schedule Visualization Tool
===========================
Visualizes the timestep schedules and step sizes for different sampling strategies.
Useful for understanding the behavior of "Linear", "Cosine", and "Quadratic" step skipping.

Usage:
    python scripts/plot_schedules.py --save_path plots/schedules.png
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from source
from src.sampler import DDIMSampler

def visualize_schedules(num_timesteps: int = 1000, num_inference_steps: int = 50, save_path: str = 'schedules.png'):
    """Visualize and compare different scheduling strategies"""
    
    # Define styles for plots
    plt.style.use('bmh')
    
    # We will instantiate a dummy sampler to access the schedule logic
    # The logic is technically in _get_timesteps, which we can access
    sampler = DDIMSampler(num_timesteps=num_timesteps)
    
    schedules = {}
    for schedule_type in ['uniform', 'quadratic', 'cosine']:
        # Support both new and old API naming if needed, but DDIM.py uses specific strings
        try:
            val_schedule_type = schedule_type
            if schedule_type == 'uniform': val_schedule_type = 'uniform' 
            
            timesteps = sampler._get_timesteps(num_inference_steps, val_schedule_type)
            schedules[schedule_type] = timesteps
        except Exception as e:
            print(f"⚠️ Could not generate {schedule_type} schedule: {e}")
            continue
    
    if not schedules:
        print("❌ No schedules generated.")
        return

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, (name, timesteps) in enumerate(schedules.items()):
        # Plot timesteps
        ax_t = axes[0, idx]
        ax_t.plot(range(len(timesteps)), timesteps, 'o-', markersize=4, color='#00539B') # Duke Blue
        ax_t.set_xlabel('Step Index')
        ax_t.set_ylabel('Timestep')
        ax_t.set_title(f'{name.capitalize()} Schedule')
        ax_t.grid(True, alpha=0.3)
        
        # Plot step sizes
        ax_s = axes[1, idx]
        step_sizes = np.abs(np.diff(timesteps))
        ax_s.plot(range(len(step_sizes)), step_sizes, 'o-', markersize=4, color='#E89923') # Duke Gold (Secondary)
        ax_s.set_xlabel('Step Index')
        ax_s.set_ylabel('Step Size |Δt|')
        ax_s.set_title(f'{name.capitalize()} Step Sizes')
        ax_s.grid(True, alpha=0.3)
        
        # Print statistics to console
        print(f"\n{name.upper()} Schedule:")
        print(f"  First 5 steps: {timesteps[:5]}")
        print(f"  Last 5 steps:  {timesteps[-5:]}")
        print(f"  Mean step size: {step_sizes.mean():.2f}")
    
    plt.suptitle(f"Diffusion Sampling Schedules (T={num_timesteps} -> {num_inference_steps} steps)", fontsize=16)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Schedule visualization saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize Diffusion Schedules')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Total timesteps')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Inference steps')
    parser.add_argument('--save_path', type=str, default='plots/schedules.png', help='Output path')
    
    args = parser.parse_args()
    
    visualize_schedules(args.num_timesteps, args.num_inference_steps, args.save_path)

if __name__ == "__main__":
    main()
