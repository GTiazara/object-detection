"""Utility functions for loading YOLO models for airplane detection."""

import sys
import pickle
import types
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class ModelLoader:
    """Handles loading and downloading YOLO models for airplane detection."""
    
    # Model configurations: (repo_id, filename, model_type)
    # model_type: "yolov8" or "yolov9" - specifies the YOLO version
    MODEL_CONFIGS = {
        # Efficient-YOLO-RS-Airplane-Detection models (YOLOv8)
        "training0": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-12/best.pt", "yolov8"),
        "training1": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-14/best.pt", "yolov8"),
        "training2": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-28/best.pt", "yolov8"),
        "training3": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-30/best.pt", "yolov8"),
        "training4": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-32/best.pt", "yolov8"),
        "training5": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-50/best.pt", "yolov8"),
        "training6": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-62/best.pt", "yolov8"),
        "training": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "training/experiment-57/best.pt", "yolov9"),
        "transfer1": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "transfer-learning/experiment-12/best.pt", "yolov8"),
        "transfer": ("iturslab/Efficient-YOLO-RS-Airplane-Detection", "transfer-learning/experiment-62/best.pt", "yolov8"),
        # Javvanny flying objects detection model (YOLOv8)
        "flying_objects": ("Javvanny/yolov8m_flying_objects_detection", "yolov8m/weights/best.pt", "yolov8"),
    }
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Directory to store downloaded models. Defaults to ~/.airplane_detection/models
        """
        if models_dir is None:
            models_dir = Path.home() / ".airplane_detection" / "models"
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
            model_type: Override model type ('yolov8' or 'yolov9'). If None, uses config default.
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
        Load a YOLO model with explicit type handling for YOLOv8 and YOLOv9.
        
        For YOLOv9 models, this handles the 'Silence' module compatibility issue
        by patching the unpickling process to map old YOLOv5 references to torch.nn.Identity.
        
        Args:
            checkpoint_path: Path to the model checkpoint file.
            model_type: Model type ('yolov8' or 'yolov9') for informational purposes.
        
        Returns:
            Loaded YOLO model instance.
        """
        if model_type == "yolov9":
            # YOLOv9 models may have references to 'models.common.Silence' which was
            # deprecated in favor of torch.nn.Identity. We need to patch the unpickling.
            return self._load_yolov9_with_patch(checkpoint_path)
        else:
            # For YOLOv8, use standard loading
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
    
    def _load_yolov9_with_patch(self, checkpoint_path: str) -> YOLO:
        """
        Load YOLOv9 model with patched unpickling to handle module compatibility.
        
        This patches torch.load to use a custom unpickler that intercepts all
        models.* module lookups and provides compatibility shims.
        
        Args:
            checkpoint_path: Path to the YOLOv9 checkpoint file.
        
        Returns:
            Loaded YOLO model instance.
        """
        # Create a Silence class that inherits from Identity for pickle compatibility
        class Silence(torch.nn.Identity):
            """Silence module compatibility shim - maps to torch.nn.Identity for YOLOv9."""
            pass
        
        # Create a comprehensive fake models package structure
        fake_models_common = types.ModuleType('models.common')
        fake_models_common.Silence = Silence
        
        fake_models_yolo = types.ModuleType('models.yolo')
        
        fake_models = types.ModuleType('models')
        fake_models.__path__ = []  # Make it a package
        fake_models.common = fake_models_common
        fake_models.yolo = fake_models_yolo
        
        # Store all original modules that might exist
        original_modules = {}
        modules_to_patch = ['models', 'models.common', 'models.yolo']
        
        for module_name in modules_to_patch:
            original_modules[module_name] = sys.modules.get(module_name)
        
        # Create a custom unpickler that handles models.* lookups
        class ModelsCompatibilityUnpickler(pickle.Unpickler):
            """Custom unpickler that provides compatibility for old models.* references."""
            
            def find_class(self, module, name):
                # Handle models.common.Silence
                if module == 'models.common' and name == 'Silence':
                    return Silence
                # Handle any other models.* lookups by returning a dummy class
                if module.startswith('models.'):
                    # Create a generic compatibility class
                    class CompatibilityClass:
                        def __init__(self, *args, **kwargs):
                            pass
                    return CompatibilityClass
                # For all other classes, use standard lookup
                return super().find_class(module, name)
        
        # Store original torch.load
        original_torch_load = torch.load
        
        def patched_torch_load(f, map_location=None, pickle_module=pickle, **kwargs):
            """Patched torch.load that uses custom unpickler for YOLOv9 compatibility."""
            if isinstance(f, (str, Path)):
                file = open(f, 'rb')
            else:
                file = f
            
            try:
                unpickler = ModelsCompatibilityUnpickler(file)
                return unpickler.load()
            finally:
                if hasattr(file, 'close') and isinstance(f, (str, Path)):
                    file.close()
        
        try:
            # Inject fake modules into sys.modules
            sys.modules['models'] = fake_models
            sys.modules['models.common'] = fake_models_common
            sys.modules['models.yolo'] = fake_models_yolo
            
            # Patch torch.load to use our custom unpickler
            torch.load = patched_torch_load
            
            # Now load the model - it will use our patched loader
            model = YOLO(checkpoint_path)
            return model
        except Exception as e:
            raise ValueError(
                f"Failed to load YOLOv9 model from {checkpoint_path}. "
                f"Error: {str(e)}\n"
                f"This may be due to compatibility issues with the checkpoint format. "
                f"Try updating ultralytics: pip install --upgrade ultralytics"
            ) from e
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
            
            # Restore original modules or remove our fake ones
            for module_name in modules_to_patch:
                if original_modules[module_name] is not None:
                    sys.modules[module_name] = original_modules[module_name]
                elif module_name in sys.modules:
                    del sys.modules[module_name]
    
    def load_local_model(self, model_path: str, model_type: Optional[str] = None) -> YOLO:
        """
        Load a YOLO model from a local file path.
        
        Args:
            model_path: Path to the local model file.
            model_type: Optional model type ('yolov8' or 'yolov9'). 
                       If None, will attempt to auto-detect from the checkpoint.
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

