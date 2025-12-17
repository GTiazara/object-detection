"""Install PyTorch with CUDA support."""

import argparse
import subprocess
import sys

# Available CUDA versions and their index URLs
CUDA_VERSIONS = {
    "cu130": "https://download.pytorch.org/whl/cu130",  # CUDA 13.0 (latest)
    "cu121": "https://download.pytorch.org/whl/cu121",  # CUDA 12.1
    "cu118": "https://download.pytorch.org/whl/cu118",  # CUDA 11.8
    "cu117": "https://download.pytorch.org/whl/cu117",  # CUDA 11.7
    "cu116": "https://download.pytorch.org/whl/cu116",  # CUDA 11.6
}

def install_pytorch_cuda(cuda_version: str = "cu130", index_url: str = None):
    """
    Install PyTorch with CUDA support.
    
    Args:
        cuda_version: CUDA version identifier (e.g., "cu130", "cu121", "cu118")
                     or "custom" if using custom index_url
        index_url: Custom PyTorch index URL (overrides cuda_version if provided)
    """
    print("=" * 60)
    print("Installing PyTorch with CUDA support")
    print("=" * 60)
    
    # Check for NVIDIA GPU
    try:
        import subprocess as sp
        result = sp.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\nNVIDIA GPU detected!")
            # Try to extract GPU name
            for line in result.stdout.split('\n'):
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                    print(f"  {line.strip()}")
                    break
        else:
            print("\nWarning: Could not detect NVIDIA GPU")
    except:
        print("\nWarning: Could not run nvidia-smi")
    
    # Determine the index URL to use
    if index_url:
        pytorch_index_url = index_url
        print(f"\nInstalling PyTorch with custom index URL: {pytorch_index_url}")
    elif cuda_version in CUDA_VERSIONS:
        pytorch_index_url = CUDA_VERSIONS[cuda_version]
        print(f"\nInstalling PyTorch with CUDA {cuda_version}...")
    else:
        print(f"\n[ERROR] Unknown CUDA version: {cuda_version}")
        print(f"Available versions: {', '.join(CUDA_VERSIONS.keys())}")
        print("Or use --index-url to specify a custom URL")
        return False
    
    print()
    
    # Uninstall current PyTorch first
    print("Uninstalling current PyTorch installation...")
    uninstall_cmd = [
        sys.executable, "-m", "pip", "uninstall", 
        "torch", "torchvision", "torchaudio", "-y"
    ]
    subprocess.call(uninstall_cmd)
    
    # Install PyTorch with specified CUDA version
    print(f"\nInstalling PyTorch with CUDA from: {pytorch_index_url}")
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", pytorch_index_url
    ]
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "=" * 60)
        print("PyTorch with CUDA installed successfully!")
        print("=" * 60)
        print("\nVerifying installation...")
        
        # Verify
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("\n[SUCCESS] CUDA is now activated!")
        else:
            print("\n[WARNING] CUDA is still not available.")
            print("You may need to:")
            print("  1. Restart Python/your IDE")
            print("  2. Update your NVIDIA drivers")
            print("  3. Check if your GPU is compatible")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Installation failed: {e}")
        print(f"\nAlternative: Install PyTorch with CUDA manually:")
        print(f"  pip install torch torchvision torchaudio --index-url {pytorch_index_url}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Install PyTorch with CUDA support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install with default CUDA 13.0 (latest)
  python install_cuda_pytorch.py
  
  # Install with CUDA 12.1
  python install_cuda_pytorch.py --cuda-version cu121
  
  # Install with CUDA 11.8
  python install_cuda_pytorch.py --cuda-version cu118
  
  # Install with custom index URL
  python install_cuda_pytorch.py --index-url https://download.pytorch.org/whl/cu130
  
Available CUDA versions: cu130, cu121, cu118, cu117, cu116
        """
    )
    parser.add_argument(
        "--cuda-version",
        type=str,
        default="cu130",
        choices=list(CUDA_VERSIONS.keys()) + ["custom"],
        help="CUDA version to install (default: cu130)"
    )
    parser.add_argument(
        "--index-url",
        type=str,
        default=None,
        help="Custom PyTorch index URL (overrides --cuda-version)"
    )
    
    args = parser.parse_args()
    
    install_pytorch_cuda(
        cuda_version=args.cuda_version,
        index_url=args.index_url
    )

