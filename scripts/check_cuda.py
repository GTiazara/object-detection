"""Check CUDA availability and configuration."""

import torch

print("=" * 60)
print("CUDA Configuration Check")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n[OK] CUDA is activated and ready to use!")
else:
    print("\nX CUDA is not available.")
    print("\nPossible reasons:")
    print("  1. PyTorch was installed without CUDA support")
    print("  2. CUDA drivers are not installed")
    print("  3. GPU is not compatible")
    print("\nTo install PyTorch with CUDA support:")
    print("  Run: python scripts/install_cuda_pytorch.py")
    print("  Or visit: https://pytorch.org/get-started/locally/")

print("=" * 60)

