# Utility Scripts

This folder contains utility scripts for managing the project.

## Scripts

### `check_cuda.py`

Check if CUDA is available and configured correctly for PyTorch.

**Usage:**
```bash
python scripts/check_cuda.py
```

Or with virtual environment:
```bash
.\venv\Scripts\activate.ps1
python scripts/check_cuda.py
```

### `install_cuda_pytorch.py`

Install PyTorch with CUDA support. This script will:
1. Detect your NVIDIA GPU
2. Uninstall the current PyTorch installation (if any)
3. Install PyTorch with CUDA 11.8 support (compatible with CUDA 11.2+ drivers)
4. Verify the installation

**Usage:**

# Install with default CUDA 13.0 (latest)
python scripts/install_cuda_pytorch.py

# Install with CUDA 12.1
python scripts/install_cuda_pytorch.py --cuda-version cu121

# Install with CUDA 11.8
python scripts/install_cuda_pytorch.py --cuda-version cu118

# Install with custom index URL
python scripts/install_cuda_pytorch.py --index-url https://download.pytorch.org/whl/cu130

```bash
python scripts/install_cuda_pytorch.py
```

Or with virtual environment:
```bash
.\venv\Scripts\activate.ps1
python scripts/install_cuda_pytorch.py
```




**Note:** Make sure your virtual environment is activated before running this script.

## CUDA Requirements

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (check with `nvidia-smi`)
- Compatible CUDA driver version (11.2 or higher recommended)

## Troubleshooting

If CUDA is not available after installation:

1. **Restart Python/IDE**: Sometimes you need to restart Python to detect CUDA
2. **Check drivers**: Run `nvidia-smi` to verify your drivers are installed
3. **Update drivers**: Visit https://www.nvidia.com/Download/index.aspx
4. **Check compatibility**: Visit https://pytorch.org/get-started/locally/ for compatible versions

