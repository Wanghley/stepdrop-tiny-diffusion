import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding injection"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.block1(x)
        
        # Add time embedding
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb
        
        h = self.block2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)
        return x + attn_out


class DownBlock(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attention=False):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attention(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res1 = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)  # concat skip
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()
    
    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attention(x)
        return x


class TinyUNet(nn.Module):
    """Improved U-Net for diffusion"""
    def __init__(self, img_size=28, channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Initial conv
        self.init_conv = nn.Conv2d(channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (Downsampling path)
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)           # 28 -> 14
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)       # 14 -> 7
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim),
        )
        
        # Decoder (Upsampling path)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, time_emb_dim)       # 7 -> 14
        self.up2 = UpBlock(base_channels * 2, base_channels, time_emb_dim)           # 14 -> 28
        
        # Output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timestep):
        # Time embedding
        t_emb = self.time_mlp(timestep)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        
        # Bottleneck (need to handle time embedding separately)
        for block in self.bottleneck:
            if isinstance(block, ResidualBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
        
        # Decoder
        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        
        # Output
        x = self.final_conv(x)
        
        return x