import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloader(batch_size, img_size, data_dir="./data"):
    """Get MNIST dataloader with proper normalization to [-1, 1]"""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader