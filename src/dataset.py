import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List
import os


class CustomImageDataset(Dataset):
    """
    Custom dataset that loads images from a folder.
    
    Expected structure:
        custom_data_dir/
            image1.png
            image2.jpg
            ...
    
    Or with subdirectories (class folders - labels ignored):
        custom_data_dir/
            class1/
                image1.png
            class2/
                image2.png
    """
    
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
    
    def __init__(
        self, 
        root_dir: str, 
        img_size: int = 64, 
        channels: int = 3,
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.channels = channels
        
        # Find all images recursively
        self.image_paths = self._find_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
        # Default transform if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5] * channels,
                    std=[0.5] * channels
                )
            ])
        else:
            self.transform = transform
    
    def _find_images(self) -> List[Path]:
        """Recursively find all image files."""
        images = []
        for ext in self.SUPPORTED_EXTENSIONS:
            images.extend(self.root_dir.rglob(f"*{ext}"))
            images.extend(self.root_dir.rglob(f"*{ext.upper()}"))
        return sorted(images)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path)
        
        # Convert to RGB or grayscale based on channels
        if self.channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        
        # Apply transforms
        image = self.transform(image)
        
        # Return image and dummy label (0)
        return image, 0


def get_dataloader(
    dataset: str = "mnist",
    batch_size: int = 128,
    img_size: int = 28,
    channels: int = 1,
    data_dir: str = "./data",
    custom_data_dir: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs
) -> DataLoader:
    """
    Get dataloader for specified dataset.
    
    Args:
        dataset: One of "mnist", "cifar10", "custom"
        batch_size: Batch size
        img_size: Target image size
        channels: Number of channels (auto-set for mnist/cifar10)
        data_dir: Directory for standard datasets
        custom_data_dir: Path to custom image folder
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader instance
    """
    
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        ds = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        actual_channels = 1
        
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        ds = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        actual_channels = 3
        
    elif dataset == "custom":
        if custom_data_dir is None:
            raise ValueError("custom_data_dir must be specified for custom dataset")
        
        ds = CustomImageDataset(
            root_dir=custom_data_dir,
            img_size=img_size,
            channels=channels
        )
        actual_channels = channels
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: mnist, cifar10, custom")
    
    print(f"Dataset: {dataset}")
    print(f"  - Samples: {len(ds)}")
    print(f"  - Image size: {img_size}x{img_size}")
    print(f"  - Channels: {actual_channels}")
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )


def get_sample_batch(dataloader: DataLoader, device: str = "cpu") -> torch.Tensor:
    """Get a single batch of images for visualization."""
    images, _ = next(iter(dataloader))
    return images.to(device)