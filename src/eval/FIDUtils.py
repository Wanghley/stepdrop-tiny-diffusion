# src/eval/FIDUtils.py
''' 
FID (Fréchet Inception Distance) helps us evaluate the generated images compared to the real CIFAR10 images.
This is done by embedding real and fake images into a feature space and using a pretrained inception network.
Each set is approximated as a Gaussian (mean + covariance).
Then, we compute the closed-form distance between the two Gaussians.
This helps us determine how close to each other they are.

'''

from typing import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths

def ensureDir(path: str) -> None:
    #Creates a dir if it doesn't exist. Ensures we don't duplicate dirs
    pathObj = Path(path)
    pathObj.mkdir(parents = True, exist_ok = True)
    return pathObj

def saveCifar10RealSub(
        outDir: str = "data/fid_real_cifar10",
        split: str = "train",
        numImages: int = 5000,
        dataDir: str = "data/cifar10",
        batchSize: int = 256,
        numWorkers: int = 4,
) -> Path:
    #Save a subset of normalized real CIFAR10 imgs as PNGS for FID reference
    outPath = ensureDir(outDir)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )
    dataset = datasets.CIFAR10(
        root = dataDir,
        train = (split == "train"),
        download = False,
        transform = transform,
    )
    loader = DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = numWorkers)
    savedCount = 0
    for batch, _ in loader:
        for image in batch:
            save_image(image, outPath / f"{savedCount:06d}.png")
            savedCount +=1
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
) -> Path:
    #Generate and save fake images to use in the FID.
    #Assume generator returns imgs in [-1,1] of shape (N,C,H,W)
    outPath = ensureDir(outDir)
    savedCount = 0
    while savedCount < numImages:
        currentBatchSize = min(batchSize, numImages-savedCount)
        images = generator(currentBatchSize).to(device)
        for image in images:
            save_image(image, outPath / f"{savedCount:06d}.png")
            savedCount += 1
    return outPath

def computeFID(
        realDir: str,
        fakeDir: str, 
        device: str = "cuda"
)->float:
    #Compute FID between two folders of images, currently set up for test
    return calculate_fid_given_paths([realDir, fakeDir], batch_size = 128, device = device, dims = 2048 )


def dummyGenerator(
        numImages: int, 
        shape = (3,32,32)
)-> torch.Tensor:
    #Dummy image generator with random noise in [-1,1].
    #Swap in sampler later
    return torch.rand((numImages, *shape)) * 2-1

"""if __name__ == "__main__":
    #Buld real cache, make dummy fake,compute FID
    realDir = saveCifar10RealSub()
    fakeDir = saveFakeImages(lambda n: dummyGenerator(n), numImages= 2000)
    fidScore = computeFID(str(realDir), str(fakeDir))
    print(f"FID (dummy vs CIFAR): {fidScore:.2f}")
"""
#Testing on cpu only rig. Takes about 10 min.
#First test -> 420.86
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    realDir = saveCifar10RealSub()
    #reduce numImages for faster tests.
    fakeDir = saveFakeImages(lambda n: dummyGenerator(n), numImages=2000, device=device)
    fidScore = computeFID(str(realDir), str(fakeDir), device=device)
    print(f"FID (dummy vs CIFAR) on {device}: {fidScore:.2f}")
