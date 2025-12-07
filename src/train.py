import torch
import torch.nn as nn
from tqdm import tqdm
from config import conf
from modules import TinyUNet
from scheduler import NoiseScheduler
from dataset import get_dataloader

def train_model():
    print(f"Training on device: {conf.device}")
    print(f"Image size: {conf.img_size}x{conf.img_size}")
    print(f"Channels: {conf.channels}")
    print(f"Base channels: {conf.base_channels}")
    print(f"Timesteps: {conf.n_timesteps}")
    
    # Initialize components
    dataloader = get_dataloader(conf.batch_size, conf.img_size)
    model = TinyUNet(
        img_size=conf.img_size, 
        channels=conf.channels, 
        base_channels=conf.base_channels
    ).to(conf.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=1e-4)
    scheduler = NoiseScheduler(
        n_timesteps=conf.n_timesteps, 
        schedule_type="cosine",
        device=conf.device
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=conf.epochs * len(dataloader), eta_min=1e-6
    )
    
    loss_fn = nn.MSELoss()
    
    # EMA for better sample quality (optional but recommended)
    ema_model = None
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(conf.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{conf.epochs}")
        epoch_loss = 0
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(conf.device)
            
            # 1. Sample random timesteps
            t = scheduler.sample_timesteps(images.shape[0])
            
            # 2. Add noise (forward diffusion)
            x_t, noise = scheduler.noise_images(images, t)
            
            # 3. Predict noise
            predicted_noise = model(x_t, t)
            
            # 4. Loss & Backprop
            loss = loss_fn(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({
                'MSE': f'{loss.item():.4f}',
                'LR': f'{lr_scheduler.get_last_lr()[0]:.6f}'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), conf.save_path)
            print(f"✅ Model saved to {conf.save_path} (loss: {avg_loss:.4f})")

    print(f"Training complete! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_model()