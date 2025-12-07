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
        
        # FIXED: Removed the "2 *" multiplier. We specify dimensions explicitly now.
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

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv2d(channels, base_channels, 3, padding=1)

        # Down Path
        self.down1 = Block(base_channels, 64, time_dim) # 32 -> 64
        self.down2 = Block(64, 128, time_dim)           # 64 -> 128
        
        # Bottleneck
        self.bot1 = Block(128, 256, time_dim)           # 128 -> 256
        
        # Up Path
        # up1 takes the bottleneck output (256) directly
        self.up1 = Block(256, 128, time_dim, up=True)   
        
        # up2 takes concatenation of up1 output (128) + down2 output (128) = 256
        self.up2 = Block(256, 64, time_dim, up=True)    
        
        self.out = nn.Conv2d(64, channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        
        d1 = self.down1(x, t)
        d2 = self.down2(d1, t)
        
        b = self.bot1(d2, t)
        
        # First Up-sample
        u1 = self.up1(b, t)
        
        # Skip Connection logic
        if u1.shape[2:] != d2.shape[2:]: 
            u1 = F.interpolate(u1, size=d2.shape[2:])
        u1 = torch.cat((u1, d2), dim=1) # 128 + 128 = 256 channels
        
        # Second Up-sample
        u2 = self.up2(u1, t)
        
        if u2.shape[2:] != d1.shape[2:]: 
            u2 = F.interpolate(u2, size=d1.shape[2:])
        u2 = torch.cat((u2, d1), dim=1)
        
        final = self.out(u2)
        if final.shape[2:] != (self.img_size, self.img_size):
            final = F.interpolate(final, size=(self.img_size, self.img_size))
        
        return final