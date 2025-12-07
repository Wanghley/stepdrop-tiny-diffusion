import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        
        # Determine if we are upsampling or just convolving
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # Inject Time Embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class TinyUNet(nn.Module):
    def __init__(self, img_size=28, channels=1, base_channels=32):
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        time_dim = 32

        # Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Initial Projection
        self.conv0 = nn.Conv2d(channels, base_channels, 3, padding=1)

        # Down Path
        self.down1 = Block(base_channels, 64, time_dim) # 32 -> 64
        self.down2 = Block(64, 128, time_dim)           # 64 -> 128
        
        # Bottleneck
        self.bot1 = Block(128, 256, time_dim)           # 128 -> 256
        
        # Up Path
        # up1 receives 256 from bot1
        self.up1 = Block(256, 128, time_dim, up=True)   
        
        # up2 receives 256 (128 from up1 + 128 from d2 skip connection)
        self.up2 = Block(256, 64, time_dim, up=True)    
        
        # Output Layer
        # FIXED: Receives 128 (64 from up2 + 64 from d1 skip connection)
        self.out = nn.Conv2d(128, channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        x0 = self.conv0(x)          # (B, 32, 28, 28)
        
        d1 = self.down1(x0, t)      # (B, 64, 14, 14)
        d2 = self.down2(d1, t)      # (B, 128, 7, 7)
        
        b = self.bot1(d2, t)        # (B, 256, 3, 3) approx
        
        # Up 1
        u1 = self.up1(b, t)
        if u1.shape[2:] != d2.shape[2:]: 
            u1 = F.interpolate(u1, size=d2.shape[2:])
        u1 = torch.cat((u1, d2), dim=1) # 128 + 128 = 256 channels
        
        # Up 2
        u2 = self.up2(u1, t)
        if u2.shape[2:] != d1.shape[2:]: 
            u2 = F.interpolate(u2, size=d1.shape[2:])
        u2 = torch.cat((u2, d1), dim=1) # 64 + 64 = 128 channels
        
        # Final Projection
        final = self.out(u2)
        
        # Ensure output size is exactly 28x28 (handling rounding errors in convolution)
        if final.shape[2:] != (self.img_size, self.img_size):
            final = F.interpolate(final, size=(self.img_size, self.img_size))
        
        return final