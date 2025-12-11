#!/usr/bin/env python3
"""
Library Check Script
====================

Verifies all required libraries are installed and checks GPU availability.
Supports: CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.
"""

import sys

def check_imports():
    """Check all required imports."""
    print("=" * 50)
    print("📦 Checking Required Libraries")
    print("=" * 50)
    
    required = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
    ]
    
    optional = [
        ("einops", "einops"),
        ("gradio", "Gradio (for GUI demo)"),
        ("pytorch_fid", "pytorch-fid (for FID)"),
        ("lpips", "LPIPS"),
        ("torchmetrics", "TorchMetrics"),
        ("cleanfid", "clean-fid"),
    ]
    
    all_ok = True
    
    print("\n🔴 Required packages:")
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - pip install {module}")
            all_ok = False
    
    print("\n🟡 Optional packages:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚪ {name} (not installed)")
    
    return all_ok


def check_device():
    """Check available compute devices."""
    import torch
    
    print("\n" + "=" * 50)
    print("🖥️  Checking Compute Devices")
    print("=" * 50)
    
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Check CUDA (NVIDIA)
    print("\n📊 CUDA (NVIDIA GPU):")
    if torch.cuda.is_available():
        print(f"  ✅ Available")
        print(f"  📍 Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  🎮 GPU {i}: {props.name} ({mem_gb:.1f} GB)")
    else:
        print(f"  ❌ Not available")
    
    # Check MPS (Apple Silicon)
    print("\n🍎 MPS (Apple Silicon GPU):")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  ✅ Available")
        if torch.backends.mps.is_built():
            print(f"  📍 MPS backend is built into PyTorch")
        # Test MPS functionality
        try:
            test_tensor = torch.zeros(1, device='mps')
            print(f"  🧪 MPS tensor test: PASSED")
        except Exception as e:
            print(f"  ⚠️ MPS tensor test failed: {e}")
    else:
        print(f"  ❌ Not available")
        if sys.platform == 'darwin':
            print(f"  💡 Tip: Update PyTorch with: pip install --upgrade torch torchvision")
    
    # Determine best device
    print("\n🎯 Recommended Device:")
    device = get_best_device()
    print(f"  → {device}")
    
    return device


def get_best_device():
    """Get the best available device."""
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def run_quick_test():
    """Run a quick tensor operation test on the best device."""
    import torch
    
    print("\n" + "=" * 50)
    print("🧪 Quick Performance Test")
    print("=" * 50)
    
    device = get_best_device()
    print(f"\nRunning on: {device}")
    
    # Matrix multiplication test
    import time
    
    size = 2000
    print(f"Testing {size}x{size} matrix multiplication...")
    
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    _ = torch.mm(a, b)
    
    # Sync before timing
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    
    start = time.time()
    for _ in range(10):
        c = torch.mm(a, b)
    
    # Sync after
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    
    print(f"  10 iterations: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms per iteration)")
    print(f"  ✅ Test passed!")


def main():
    print("\n" + "🔍 " + "=" * 46 + " 🔍")
    print("      StepDrop Environment Check")
    print("🔍 " + "=" * 46 + " 🔍\n")
    
    imports_ok = check_imports()
    device = check_device()
    
    if imports_ok:
        run_quick_test()
    
    print("\n" + "=" * 50)
    print("📋 Summary")
    print("=" * 50)
    
    if imports_ok:
        print("  ✅ All required libraries installed")
    else:
        print("  ❌ Some required libraries missing")
    
    print(f"  🖥️  Best device: {device}")
    
    if device == "cpu":
        print("\n  ⚠️  Running on CPU will be slow!")
        print("  💡 For better performance:")
        print("     - NVIDIA GPU: Install CUDA toolkit")
        print("     - Apple Silicon: pip install --upgrade torch")
    
    print("\n" + "=" * 50)
    print("🚀 Ready to run StepDrop!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

