"""Utility functions for loading YOLO models."""

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class ModelLoader:
    """Handles loading and downloading YOLO models."""
    
    # Model configurations: (repo_id, filename, model_type)
    # model_type: "yolov8" - specifies the YOLO version
    MODEL_CONFIGS = {
        # Efficient-YOLO-RS-Airplane-Detection models (YOLOv8)
        "training0": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-12/best.pt", "yolov8"),
        "training1": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-14/best.pt", "yolov8"),
        "training2": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-28/best.pt", "yolov8"),
        "training3": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-30/best.pt", "yolov8"),
        "training4": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-32/best.pt", "yolov8"),
        "training5": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-50/best.pt", "yolov8"),
        "training6": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-62/best.pt", "yolov8"),
        "training": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-57/best.pt", "yolov8"),
        "transfer1": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "transfer-learning/experiment-12/best.pt", "yolov8"),
        "transfer": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "transfer-learning/experiment-62/best.pt", "yolov8"),
        # Javvanny flying objects detection model (YOLOv8)
        "flying_objects": ("Javvanny/yolov8m_flying_objects_detection", "yolov8m/weights/best.pt", "yolov8"),
        "flying_airplane": ("keremberke/yolov8m-plane-detection", "best.pt", "yolov8"),
    }
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Directory to store downloaded models. Defaults to ~/.yolo_detection/models
        """
        if models_dir is None:
            models_dir = Path.home() / ".yolo_detection" / "models"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_from_hf(self, repo_id: str, filename: str, force_download: bool = False) -> Path:
        """Download the specified checkpoint from Hugging Face and return the local path."""
        print(f"Downloading '{filename}' from Hugging Face repository '{repo_id}'…")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self.models_dir),
            local_dir_use_symlinks=False,
            force_download=force_download,
        )
        print(f"Checkpoint available at {local_path}")
        return Path(local_path)

    def load_model(
        self,
        model_variant: str = "training",
        *,
        local_path: Optional[str] = None,
        force_download: bool = False,
        model_type: Optional[str] = None,
    ) -> YOLO:
        """
        Load a YOLO model either from Hugging Face or a user-provided local path.

        Args:
            model_variant: Selects which checkpoint to download ('training' or 'transfer').
            local_path: If provided, bypass download and load this file directly.
            force_download: Redownload checkpoint even if it exists locally.
            model_type: Override model type ('yolov8'). If None, uses config default.
        """
        if local_path:
            return self.load_local_model(local_path, model_type=model_type)

        if model_variant not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown variant '{model_variant}'. Available options: {list(self.MODEL_CONFIGS.keys())}"
            )

        repo_id, filename, default_model_type = self.MODEL_CONFIGS[model_variant]
        checkpoint_path = self._download_from_hf(repo_id, filename, force_download=force_download)

        # Use provided model_type or fall back to config default
        model_type_to_use = model_type if model_type is not None else default_model_type

        print(f"Loading {model_type_to_use} model from {checkpoint_path}…")
        model = self._load_yolo_model(str(checkpoint_path), model_type_to_use)
        print("Model loaded successfully! You can now call model.predict(...) as in Ultralytics docs.")
        return model

    def _load_yolo_model(self, checkpoint_path: str, model_type: str) -> YOLO:
        """
        Load a YOLO model.
        
        Args:
            checkpoint_path: Path to the model checkpoint file.
            model_type: Model type ('yolov8') for informational purposes.
        
        Returns:
            Loaded YOLO model instance.
        """
        try:
            model = YOLO(checkpoint_path)
            return model
        except Exception as e:
            raise ValueError(
                f"Failed to load {model_type} model from {checkpoint_path}. "
                f"Error: {str(e)}\n"
                f"Note: Make sure you have the latest version of ultralytics installed: "
                f"pip install --upgrade ultralytics"
            ) from e
    
    def load_local_model(self, model_path: str, model_type: Optional[str] = None) -> YOLO:
        """
        Load a YOLO model from a local file path.
        
        Args:
            model_path: Path to the local model file.
            model_type: Optional model type ('yolov8'). 
                       If None, defaults to yolov8.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # If model_type is not provided, try to detect it or default to yolov8
        if model_type is None:
            # Try to infer from filename or default to yolov8
            # You could add logic here to detect from checkpoint metadata if needed
            model_type = "yolov8"
            print(f"Model type not specified, defaulting to {model_type}")

        print(f"Loading {model_type} model from {path}…")
        model = self._load_yolo_model(str(path), model_type)
        print("Model loaded successfully! You can now call model.predict(...) as in Ultralytics docs.")
        return model

