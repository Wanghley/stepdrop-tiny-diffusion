import torchvision.datasets as datasets

def downloadCifar10(rootDir: str = "data/cifar10") -> None:
    _ = datasets.CIFAR10(root=rootDir, train=True, download=True)
    _ = datasets.CIFAR10(root=rootDir, train=False, download=True)
    print(f"✅ CIFAR-10 downloaded (or already present) at: {rootDir}")

if __name__ == "__main__":
    downloadCifar10()