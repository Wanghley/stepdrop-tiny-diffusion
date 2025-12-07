import torch
import torch.nn as nn
from tqdm import tqdm
from config import conf
from modules import TinyUNet
from scheduler import NoiseScheduler
from dataset import get_dataloader

def train_model():
    print(f"Training on device: {conf.device}")
    
    # Initialize components
    dataloader = get_dataloader(conf.batch_size, conf.img_size)
    model = TinyUNet(img_size=conf.img_size, channels=conf.channels, base_channels=conf.base_channels).to(conf.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    scheduler = NoiseScheduler(n_timesteps=conf.n_timesteps, device=conf.device)
    loss_fn = nn.MSELoss()

    model.train()
    
    for epoch in range(conf.epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0
        for i, (images, _) in enumerate(pbar):
            images = images.to(conf.device)
            
            # 1. Sample random timesteps
            t = scheduler.sample_timesteps(images.shape[0])
            
            # 2. Add noise
            x_t, noise = scheduler.noise_images(images, t)
            
            # 3. Predict noise
            predicted_noise = model(x_t, t)
            
            # 4. Loss & Backprop
            loss = loss_fn(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(dataloader):.4f}")

    # Save Model
    torch.save(model.state_dict(), conf.save_path)
    print(f"Model saved to {conf.save_path}")

if __name__ == "__main__":
    train_model()