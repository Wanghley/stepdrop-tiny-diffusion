import torch

class Config:
    def __init__(self):
        """
        Configuration settings for the diffusion model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 28
        self.channels = 1
        self.batch_size = 128
        self.n_timesteps = 1000 
        self.lr = 2e-4
        self.epochs = 20
        self.base_channels = 64
        self.save_path = "tiny_ddpm.pt"
        
        # Beta schedule parameters
        self.beta_start = 1e-4
        self.beta_end = 0.02

conf = Config()