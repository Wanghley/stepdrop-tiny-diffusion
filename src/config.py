import torch

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 28
        self.channels = 1        # MNIST is grayscale
        self.batch_size = 128
        self.n_timesteps = 1000  # Standard T
        self.lr = 2e-4
        self.epochs = 5          # Increase this to 20+ for high quality
        self.base_channels = 32  # Width of the tiny U-Net
        self.save_path = "tiny_ddpm.pt"

conf = Config()