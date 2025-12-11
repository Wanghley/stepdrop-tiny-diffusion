from scripts.demo_gui import create_demo, get_best_device

# Use the default checkpoint (adjust path if needed)
checkpoint_path = "checkpoints/cifar10_64ch_50ep.pt"
device = get_best_device()

app = create_demo(checkpoint_path, device)