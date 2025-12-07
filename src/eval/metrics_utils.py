# src/eval/metrics_utils.py
''' 
Metrics Utility Module

This module provides a unified interface for calculating standard generative model metrics.
It follows the "camelCase" convention as requested.

Supported Metrics:
1.  **FID (Fréchet Inception Distance)**: 
    Measures the similarity between two datasets of images (real vs. generated).
    Lower is better. It compares the statistics (mean and covariance) of the Inception-v3 pool3 features.
    
2.  **IS (Inception Score)**: 
    Measures quality and diversity of generated images.
    Higher is better. It uses the Inception-v3 network to calculate the KL divergence between 
    the conditional class distribution p(y|x) and the marginal class distribution p(y).
    
3.  **FLOPs (Floating Point Operations)**:
    Measures the theoretical computational cost of the model.
    Lower is better for efficiency.

'''

from typing import Callable, Tuple, List, Optional
from pathlib import Path
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics.image.inception import InceptionScore
from thop import profile, clever_format

def ensureDir(path: str) -> Path:
    """Creates a dir if it doesn't exist. Ensures we don't duplicate dirs."""
    pathObj = Path(path)
    pathObj.mkdir(parents=True, exist_ok=True)
    return pathObj

def saveCifar10RealSub(
        outDir: str = "data/fid_real_cifar10",
        split: str = "train",
        numImages: int = 5000,
        dataDir: str = "data/cifar10",
        batchSize: int = 256,
        numWorkers: int = 4,
) -> Path:
    """
    Save a subset of normalized real CIFAR10 imgs as PNGS for FID reference.
    We need this because pytorch-fid reads from folders.
    """
    outPath = ensureDir(outDir)
    
    # Check if already generated enough images
    num_existing = len(list(outPath.glob("*.png")))
    if num_existing >= numImages:
        print(f"Dataset subset already exists at {outPath} ({num_existing} images)")
        return outPath

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = datasets.CIFAR10(
        root=dataDir,
        train=(split == "train"),
        download=False,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)
    savedCount = 0
    
    print(f"Generating {numImages} real images to {outPath}...")
    for batch, _ in loader:
        for image in batch:
            save_image(image, outPath / f"{savedCount:06d}.png")
            savedCount += 1
            if savedCount >= numImages:
                return outPath
    return outPath

@torch.no_grad()
def saveFakeImages(
    generator: Callable[[int], torch.Tensor],
    outDir: str = "data/fid_fake_samples",
    numImages: int = 5000,
    batchSize: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    clearDir: bool = True
) -> Path:
    """
    Generate and save fake images to use in the FID.
    Assume generator returns imgs in [-1, 1] of shape (N, C, H, W).
    """
    outPath = ensureDir(outDir)
    
    # Optionally clear existing images to prevent stale data
    if clearDir:
        for f in outPath.glob("*.png"):
            f.unlink()
            
    savedCount = 0
from tqdm import tqdm

# ... (imports)

# ...

@torch.no_grad()
def saveFakeImages(
    generator: Callable[[int], torch.Tensor],
    outDir: str = "data/fid_fake_samples",
    numImages: int = 5000,
    batchSize: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    clearDir: bool = True
) -> Path:
    """
    Generate and save fake images to use in the FID.
    Assume generator returns imgs in [-1, 1] of shape (N, C, H, W).
    """
    outPath = ensureDir(outDir)
    
    # Optionally clear existing images to prevent stale data
    if clearDir:
        for f in outPath.glob("*.png"):
            f.unlink()
            
    savedCount = 0
    # print(f"Generating {numImages} fake images to {outPath}...")
    
    pbar = tqdm(total=numImages, desc=f"Generating to {outPath.name}", unit="img")
    
    while savedCount < numImages:
        currentBatchSize = min(batchSize, numImages - savedCount)
        images = generator(currentBatchSize).to(device)
        
        # Ensure images are in proper range for saving [-1, 1] -> [0, 1]
        for image in images:
            save_image(image, outPath / f"{savedCount:06d}.png", normalize=True, value_range=(-1, 1))
            savedCount += 1
            pbar.update(1)
            
    pbar.close()
    return outPath
            
    return outPath

def computeFID(
        realDir: str,
        fakeDir: str, 
        device: str = "cuda"
) -> float:
    """
    Compute FID between two folders of images.
    
    FID encapsulates the distance between the distribution of real and generated images.
    We compute the pool3 features of InceptionV3 for both sets, then calculate:
    FID = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    """
    # pytorch-fid takes a list of paths
    return calculate_fid_given_paths([realDir, fakeDir], batch_size=128, device=device, dims=2048)

def computeInceptionScore(
    fakeDir: str,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS) for a directory of images.
    
    IS evaluates:
    1. Clarity: Do images look like a specific object? (Low entropy p(y|x))
    2. Diversity: Do images cover many classes? (High entropy p(y))
    
    Returns (mean_is, std_is)
    """
    # Prepare standard torchmetrics IS
    # It expects images normalized to [0, 255] byte type usually for compatibility
    # But Custom path loading is needed.
    
    dataset = datasets.ImageFolder(root=str(Path(fakeDir).parent), transform=transforms.ToTensor())
    # Note: ImageFolder expects subdirectories. If fakeDir is just images, we might use a different wrapper.
    # To be simpler, let's just load the images manually into a tensor since we know where they are.
    
    pathObj = Path(fakeDir)
    files = list(pathObj.glob("*.png"))
    if not files:
        return 0.0, 0.0
    
    # Load all images into memory (batched) to avoid OOM
    # For 32x32 CIFAR size, we can load a fair amount.
    all_imgs = []
    
    # We load standard [0,1] images. torchmetrics IS expects [0,255] uint8 usually for best compliance with strict implementations
    # or [0,1] floats. Let's use [0,255] uint8 as it's standard for IS papers.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte()) # Convert to uint8
    ])
    
    # Manual loader
    print("Loading images for IS calculation...")
    imgs_tensor = []
    for f in files:
        img = datasets.folder.default_loader(f) # Utils for loading
        t_img = transform(img)
        imgs_tensor.append(t_img)
        
    imgs_tensor = torch.stack(imgs_tensor).to(device)
    
    inception = InceptionScore().to(device)
    inception.update(imgs_tensor)
    mean, std = inception.compute()
    
    return mean.item(), std.item()

def computeFLOPs(
    model: nn.Module,
    sampleShape: Tuple[int, int, int, int], # (1, C, H, W)
    device: str = "cuda"
) -> float:
    """
    Compute FLOPs for a SINGLE Step of the model.
    To get total inference FLOPs, we multiply this by the number of steps.
    """
    # Create dummy inputs
    inputs = torch.randn(sampleShape).to(device)
    timesteps = torch.tensor([0]).to(device) # Single timestep
    
    # thop profile
    flops, params = profile(model, inputs=(inputs, timesteps), verbose=False)
    
    return flops


# Dummy generator for testing
def dummyGenerator(
        numImages: int, 
        shape=(3, 32, 32)
) -> torch.Tensor:
    return torch.rand((numImages, *shape)) * 2 - 1

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Metrics Utils on {device}...")
    
    # 1. Setup Data
    realDir = saveCifar10RealSub(numImages=100) # Small for test
    fakeDir = saveFakeImages(lambda n: dummyGenerator(n), numImages=100, device=device)
    
    # 2. Test FID
    try:
        fidScore = computeFID(str(realDir), str(fakeDir), device=device)
        print(f"✅ FID (Dummy): {fidScore:.2f}")
    except Exception as e:
        print(f"❌ FID Failed: {e}")
        
    # 3. Test IS
    try:
        isMean, isStd = computeInceptionScore(str(fakeDir), device=device)
        print(f"✅ IS (Dummy): {isMean:.2f} ± {isStd:.2f}")
    except Exception as e:
        print(f"❌ IS Failed: {e}")
        
    # 4. Test FLOPs (Mock Model)
    class MockModel(nn.Module):
        def forward(self, x, t):
            return x * t
            
    try:
        model = MockModel().to(device)
        flops = computeFLOPs(model, (1, 3, 32, 32), device=device)
        print(f"✅ FLOPs (Dummy): {flops}")
    except Exception as e:
         print(f"❌ FLOPs Failed: {e}")
