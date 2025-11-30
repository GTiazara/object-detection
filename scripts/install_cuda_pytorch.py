"""Install PyTorch with CUDA support."""

import subprocess
import sys

def install_pytorch_cuda():
    """Install PyTorch with CUDA 11.8 support (compatible with CUDA 11.2+ drivers)."""
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
    
    print("\nInstalling PyTorch with CUDA 11.8 (compatible with CUDA 11.2+ drivers)...")
    print()
    
    # Uninstall current PyTorch first
    print("Uninstalling current PyTorch installation...")
    uninstall_cmd = [
        sys.executable, "-m", "pip", "uninstall", 
        "torch", "torchvision", "torchaudio", "-y"
    ]
    subprocess.call(uninstall_cmd)
    
    # PyTorch with CUDA 11.8 (compatible with CUDA 11.2 drivers)
    print("\nInstalling PyTorch with CUDA 11.8...")
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
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
        print("\nAlternative: Install PyTorch with CUDA 11.8 manually:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    install_pytorch_cuda()

