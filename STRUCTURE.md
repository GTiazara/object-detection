# Project Structure

## Current Clean Structure

```
src/
├── detector.py              # Core detection functionality
├── model_loader.py          # Model loading and management
├── predict/                 # Prediction modules
│   ├── h5.py               # H5/HDF5 file prediction (renamed from predict_h5.py)
│   ├── consumer.py         # Queue-based consumer for batch processing
│   ├── main.py             # Main CLI for image file prediction
│   └── predict_example.py  # Example scripts
├── training/               # Training modules
│   ├── train.py
│   └── trainer.py
└── utils/                   # Shared utilities (reusable across modules)
    ├── tile.py             # Image tiling and merging functionality
    └── queue_manager.py    # Queue management utilities
```

## Improvements Made

1. **Created `utils/` folder**: Moved reusable utilities here
   - `tile.py`: Generic tiling functionality (used by H5 and can be used by other modules)
   - `queue_manager.py`: Queue management (used by consumer)

2. **Renamed `predict_h5.py` to `h5.py`**: Shorter, cleaner name

3. **Better organization**: 
   - Core functionality at root level (`detector.py`, `model_loader.py`)
   - Prediction modules in `predict/`
   - Shared utilities in `utils/`
   - Training modules in `training/`

## Import Paths

### Updated Imports
- `from src.utils.tile import ...` (was `from src.predict.tile import ...`)
- `from src.utils.queue_manager import QueueManager` (was `from src.queue_manager import ...`)
- `python -m src.predict.h5` (was `python -m src.predict.predict_h5`)

### Unchanged Imports
- `from src.detector import Detector`
- `from src.model_loader import ModelLoader`
- `from src.predict.main import main`
- `from src.predict.consumer import ...`

## Benefits

1. **Clear separation**: Utilities are separated from domain-specific code
2. **Reusability**: `utils/` contains components that can be used across the project
3. **Maintainability**: Related functionality is grouped together
4. **Scalability**: Easy to add new prediction types (e.g., `predict/video.py`, `predict/batch.py`)

