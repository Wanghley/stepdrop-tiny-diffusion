import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class Config:
    """Configuration for diffusion model training and sampling."""
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    dataset: str = "mnist"  # mnist, cifar10, custom
    data_dir: str = "./data"
    custom_data_dir: Optional[str] = None  # For custom datasets
    img_size: int = 28
    channels: int = 1
    
    # Model
    base_channels: int = 64
    time_emb_dim: int = 128
    
    # Training
    batch_size: int = 128
    n_timesteps: int = 1000
    lr: float = 2e-4
    epochs: int = 20
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    save_every: int = 5  # Save checkpoint every N epochs
    
    # Scheduler
    schedule_type: str = "cosine"  # linear, cosine
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Paths
    save_path: str = "checkpoints/model.pt"
    log_dir: str = "logs"
    sample_dir: str = "samples"
    
    # Sampling
    n_samples: int = 16
    sampling_method: str = "ddpm"  # ddpm, ddim
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    
    # Resume training
    resume: Optional[str] = None  # Path to checkpoint to resume from
    
    # Misc
    seed: Optional[int] = None
    num_workers: int = 4
    
    def __post_init__(self):
        """Auto-configure based on dataset."""
        if self.dataset == "mnist":
            self.img_size = 28
            self.channels = 1
        elif self.dataset == "cifar10":
            self.img_size = 32
            self.channels = 3
        # custom dataset keeps user-specified values
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create config from argparse namespace."""
        return cls(**{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    def get_device(self) -> torch.device:
        return torch.device(self.device)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common arguments to any parser."""
    # Data args
    parser.add_argument("--dataset", type=str, default="mnist", 
                        choices=["mnist", "cifar10", "custom"],
                        help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory for dataset storage")
    parser.add_argument("--custom_data_dir", type=str, default=None,
                        help="Path to custom image folder dataset")
    parser.add_argument("--img_size", type=int, default=28,
                        help="Image size (height=width)")
    parser.add_argument("--channels", type=int, default=1,
                        help="Number of image channels")
    
    # Model args
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Base channel count for U-Net")
    parser.add_argument("--time_emb_dim", type=int, default=128,
                        help="Time embedding dimension")
    
    # Scheduler args
    parser.add_argument("--n_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--schedule_type", type=str, default="cosine",
                        choices=["linear", "cosine"],
                        help="Noise schedule type")
    
    # Paths
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pt",
                        help="Path to save/load model")
    
    # Device
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    return parser


# Global config instance (can be overridden)
conf = Config()