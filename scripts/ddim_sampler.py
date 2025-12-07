import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

class Scheduler:
    
    @staticmethod
    def get_schedule(schedule_type: str, num_timesteps: int, num_inference_steps: int) -> np.ndarray:
        """
        Returns:
            Array of timesteps from high to low (for sampling T -> 0)
        """
        if schedule_type == 'uniform':
            return Scheduler._uniform_schedule(num_timesteps, num_inference_steps)
        elif schedule_type == 'quadratic':
            return Scheduler._quadratic_schedule(num_timesteps, num_inference_steps)
        elif schedule_type == 'cosine':
            return Scheduler._cosine_schedule(num_timesteps, num_inference_steps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    @staticmethod
    def _uniform_schedule(num_timesteps: int, num_inference_steps: int) -> np.ndarray:
        step_ratio = num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        return timesteps
    
    @staticmethod
    def _quadratic_schedule(num_timesteps: int, num_inference_steps: int) -> np.ndarray:
        t_normalized = np.linspace(0, 1, num_inference_steps)
        timesteps = (num_timesteps - 1) * (1 - t_normalized) ** 2
        return timesteps.round().astype(np.int64)
    
    @staticmethod
    def _cosine_schedule(num_timesteps: int, num_inference_steps: int) -> np.ndarray:
        t_normalized = np.linspace(0, 1, num_inference_steps)
        # cos(t * π/2) decreases from 1 to 0
        timesteps = (num_timesteps - 1) * (np.cos(t_normalized * np.pi / 2)**2)
        return timesteps.round().astype(np.int64)

class DDIMSampler:
    """Denoising Diffusion Implicit Models Sampling"""
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        self.betas = self._cosine_beta_schedule(num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _estimate_error(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> float:
        """
        Estimate reconstruction error useing prediction stability
        """
        with torch.no_grad():
            pred_noise_1 = model(x, t)
            
            # Add small perturbation and predict again
            noise_scale = 0.01
            perturbation = torch.randn_like(x) * noise_scale
            pred_noise_2 = model(x + perturbation, t)
            error = (pred_noise_1 - pred_noise_2).abs().mean().item()
        return error
    
    def _compute_adaptive_step_size(self, error: float, min_step: int = 5, max_step: int = 50,
        error_threshold_low: float = 0.05,
        error_threshold_high: float = 0.15
    ) -> int:
        """
        Map error to step size: 
        High error → small steps (careful denoising)
        Low error → large steps (can skip)
        """
        if error > error_threshold_high:
            return min_step
        elif error < error_threshold_low:
            return max_step
        else:
            # Linear interpolation
            error_normalized = (error - error_threshold_low) / \
                             (error_threshold_high - error_threshold_low)
            step_size = max_step - error_normalized * (max_step - min_step)
            return int(step_size)
    
    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple, num_inference_steps: int = 50, eta: float = 0.0, 
        device: str = "cuda", 
        return_all_timesteps: bool = False,
        schedule_type: str = "uniform",
        adaptive_params: Optional[dict] = None
    ):
        """
        Returns:
            Generated samples (and optionally all intermediate steps)
        """
        batch_size = shape[0]
        
        if schedule_type == 'adaptive':
            timesteps = None
            if adaptive_params is None:
                adaptive_params = {
                    'min_step': 5,
                    'max_step': 50,
                    'error_threshold_low': 0.05,
                    'error_threshold_high': 0.15
                }
        else:
            timesteps = Scheduler.get_schedule(
                schedule_type, 
                self.num_timesteps, 
                num_inference_steps
            )
            timesteps = torch.from_numpy(timesteps).to(device)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        all_samples = [x] if return_all_timesteps else None
        timesteps_used = []  
        errors_recorded = []  # adaptive only
        
        # Iteratively denoise
        if schedule_type == 'adaptive':
            current_t = self.num_timesteps - 1
            step_count = 0
            
            while current_t > 0 and step_count < num_inference_steps:
                t_batch = torch.full((batch_size,), current_t, device=device, dtype=torch.long)
                timesteps_used.append(current_t)
                
                # Estimate error
                error = self._estimate_error(model, x, t_batch)
                errors_recorded.append(error)
                
                # Predict noise
                predicted_noise = model(x, t_batch)
                
                # Compute step size based on error
                step_size = self._compute_adaptive_step_size(error, **adaptive_params)
                next_t = max(0, current_t - step_size)
                
                # Get alpha values
                alpha_cumprod_t = self.alphas_cumprod[current_t].to(device)
                alpha_cumprod_t_prev = self.alphas_cumprod[next_t].to(device) if next_t > 0 else torch.tensor(1.0).to(device)
                
                # Predict x_0
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)
                
                # Compute variance
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                    (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                )
                
                # Compute direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise
                
                # Add noise
                if next_t > 0 and eta > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Compute x_{t-1}
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma_t * noise
                
                if return_all_timesteps:
                    all_samples.append(x)
                
                current_t = next_t
                step_count += 1
        
        else:
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                timesteps_used.append(t.item())
                
                predicted_noise = model(x, t_batch)
                alpha_cumprod_t = self.alphas_cumprod[t].to(device)
                
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)
                
                # Get previous timestep
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_cumprod_t_prev = self.alphas_cumprod[t_prev].to(device)
                else:
                    alpha_cumprod_t_prev = torch.tensor(1.0).to(device)
                
                # Compute variance
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                    (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                )
                
                # Compute direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise
                
                # Add noise
                if i < len(timesteps) - 1 and eta > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Compute x_{t-1}
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma_t * noise
                
                if return_all_timesteps:
                    all_samples.append(x)
        
        # Prepare return value
        result = {
            'samples': torch.stack(all_samples) if return_all_timesteps else x,
            'timesteps_used': timesteps_used,
        }
        
        if schedule_type == 'adaptive':
            result['errors'] = errors_recorded
        
        return result


def visualize_schedules(num_timesteps: int = 1000, num_inference_steps: int = 50, save_path: str = 'schedules.png'):
    """Visualize and compare different scheduling strategies"""
    import matplotlib.pyplot as plt
    
    schedules = {}
    for schedule_type in ['uniform', 'quadratic', 'cosine']:
        timesteps = Scheduler.get_schedule(schedule_type, num_timesteps, num_inference_steps)
        schedules[schedule_type] = timesteps
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, (name, timesteps) in enumerate(schedules.items()):
        # Plot timesteps
        axes[0, idx].plot(range(len(timesteps)), timesteps, 'o-', markersize=4)
        axes[0, idx].set_xlabel('Step Index')
        axes[0, idx].set_ylabel('Timestep')
        axes[0, idx].set_title(f'{name.capitalize()} Schedule')
        axes[0, idx].grid(True, alpha=0.3)
        
        # Plot step sizes
        step_sizes = np.abs(np.diff(timesteps))
        axes[1, idx].plot(range(len(step_sizes)), step_sizes, 'o-', markersize=4)
        axes[1, idx].set_xlabel('Step Index')
        axes[1, idx].set_ylabel('Step Size |Δt|')
        axes[1, idx].set_title(f'{name.capitalize()} Step Sizes')
        axes[1, idx].grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\n{name.upper()} Schedule:")
        print(f"  Timesteps: {timesteps[:5]} ... {timesteps[-5:]}")
        print(f"  First 5 step sizes: {step_sizes[:5]}")
        print(f"  Last 5 step sizes: {step_sizes[-5:]}")
        print(f"  Mean step size: {step_sizes.mean():.2f}")
        print(f"  Std step size: {step_sizes.std():.2f}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSchedule visualization saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='DDIM Sampling with Different Schedules')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Total number of diffusion timesteps')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of sampling steps (accelerated)')
    parser.add_argument('--eta', type=float, default=0.0, help='Stochasticity parameter (0=deterministic, 1=stochastic like DDPM)')
    parser.add_argument('--output_dir', type=str, default='./samples', help='Directory to save samples')
    parser.add_argument('--return_all', action='store_true', help='Return all timesteps')
    parser.add_argument('--schedule', type=str, default='uniform', 
                       choices=['uniform', 'quadratic', 'cosine', 'adaptive'],
                       help='Timestep scheduling strategy')
    parser.add_argument('--visualize_schedules', action='store_true', 
                       help='Visualize schedule comparison and exit')
    
    # Adaptive scheduling parameters
    parser.add_argument('--min_step', type=int, default=5, help='Minimum step size for adaptive scheduling')
    parser.add_argument('--max_step', type=int, default=50, help='Maximum step size for adaptive scheduling')
    parser.add_argument('--error_threshold_low', type=float, default=0.05, help='Low error threshold')
    parser.add_argument('--error_threshold_high', type=float, default=0.15, help='High error threshold')
    
    args = parser.parse_args()
    
    # If visualization requested, just show schedules and exit
    if args.visualize_schedules:
        print("Generating schedule visualizations...")
        visualize_schedules(args.num_timesteps, args.num_inference_steps)
        return
    
    # Hard-coded parameters
    model_path = 'checkpoints/model.pt'
    image_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Initialize sampler
    print(f"Initializing DDIM sampler with {args.num_timesteps} total timesteps")
    print(f"Using {args.num_inference_steps} inference steps (eta={args.eta})")
    print(f"Schedule type: {args.schedule}")
    sampler = DDIMSampler(num_timesteps=args.num_timesteps)
    
    # Prepare adaptive parameters if needed
    adaptive_params = None
    if args.schedule == 'adaptive':
        adaptive_params = {
            'min_step': args.min_step,
            'max_step': args.max_step,
            'error_threshold_low': args.error_threshold_low,
            'error_threshold_high': args.error_threshold_high
        }
        print(f"Adaptive parameters: {adaptive_params}")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    shape = (args.num_samples, 3, image_size, image_size)
    
    import time
    start_time = time.time()
    
    result = sampler.sample(
        model=model,
        shape=shape,
        num_inference_steps=args.num_inference_steps,
        eta=args.eta,
        device=device,
        return_all_timesteps=args.return_all,
        schedule_type=args.schedule,
        adaptive_params=adaptive_params
    )
    
    sampling_time = time.time() - start_time
    
    samples = result['samples']
    timesteps_used = result['timesteps_used']
    
    # Print statistics
    print(f"\nSampling completed in {sampling_time:.2f}s")
    print(f"Actual steps used: {len(timesteps_used)}")
    print(f"Time per step: {sampling_time / len(timesteps_used):.3f}s")
    
    if args.schedule == 'adaptive':
        errors = result['errors']
        print(f"Error statistics:")
        print(f"  Mean: {np.mean(errors):.4f}")
        print(f"  Std: {np.std(errors):.4f}")
        print(f"  Min: {np.min(errors):.4f}")
        print(f"  Max: {np.max(errors):.4f}")
    
    # Save samples
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'ddim_samples_{args.schedule}_t{args.num_timesteps}_steps{args.num_inference_steps}_eta{args.eta}.pt'
    
    # Save all results
    save_dict = {
        'samples': samples,
        'timesteps_used': timesteps_used,
        'schedule_type': args.schedule,
        'num_inference_steps': args.num_inference_steps,
        'sampling_time': sampling_time,
    }
    
    if args.schedule == 'adaptive':
        save_dict['errors'] = errors
        save_dict['adaptive_params'] = adaptive_params
    
    torch.save(save_dict, output_path)
    print(f"Results saved to {output_path}")
    print(f"Sample shape: {samples.shape}")

if __name__ == "__main__":
    main()