"""Fine-tuning module for YOLOv8 models with custom datasets."""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import yaml

from ultralytics import YOLO
from pathlib import Path
from src.model_loader import ModelLoader

from clearml import Task
import inspect
import random

class FineTuner:
    """Fine-tune YOLOv8 models on custom datasets in YOLOv8 format."""

    def __init__(
        self,
        base_model_variant: str = "training0",
        base_model_path: Optional[str] = None,
        models_dir: Optional[str] = None,
    ):
        """
        Initialize the fine-tuner.

        Args:
            base_model_variant: Model variant to use as base (e.g., "training0")
            base_model_path: Optional local path to base model (overrides variant)
            models_dir: Directory to store/download models
        """
        self.base_model_variant = base_model_variant
        self.base_model_path = base_model_path
        self.models_dir = models_dir
        self.model_loader = ModelLoader(models_dir=models_dir)

    def load_base_model(self) -> YOLO:
        """
        Load the base model for fine-tuning.

        Returns:
            Loaded YOLO model ready for training
        """
        if self.base_model_path:
            print(f"Loading base model from local path: {self.base_model_path}")
            model = self.model_loader.load_local_model(self.base_model_path)
        else:
            print(f"Loading base model variant: {self.base_model_variant}")
            model = self.model_loader.load_model(
                model_variant=self.base_model_variant, force_download=False
            )

        print("Base model loaded successfully!")
        return model

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 100,
        imgsz: int = 960,
        batch: int = 16,
        device: Optional[str] = None,
        project: str = "training/runs",
        name: str = "fine_tune",
        patience: int = 50,
        save_period: int = 10,
        use_clearml: bool = False,
        clearml_project_name: Optional[str] = None,
        clearml_task_name: Optional[str] = None,
        num_prediction_plots: int = 0,
        **kwargs,
    ) -> YOLO:
        """
        Fine-tune the model on a custom dataset.

        Args:
            dataset_yaml: Path to dataset YAML file (YOLOv8 format)
            epochs: Number of training epochs
            imgsz: Image size for training (640, 960, or 1280)
            batch: Batch size
            device: Device to use ('cuda', 'cpu', or None for auto)
            project: Project directory for saving results
            name: Experiment name
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            **kwargs: Additional training arguments (see Ultralytics docs)

        Returns:
            Trained YOLO model
        """
        # Validate dataset YAML
        dataset_path = Path(dataset_yaml)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

        # Load base model
        model = self.load_base_model()

        # Prepare training arguments
        train_args = {
            "data": str(dataset_path),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": project,
            "name": name,
            "patience": patience,
            "save_period": save_period,
            **kwargs,
        }

        # Explicitly set AMP if not provided to skip validation check that downloads yolo11n.pt
        if "amp" not in train_args:
            train_args["amp"] = True

        if device:
            train_args["device"] = device

        # Note: YOLOv8 automatically resizes images to imgsz during training
        # The imgsz parameter already handles resizing efficiently

        print(f"\nStarting fine-tuning with parameters:")
        print(f"  Dataset: {dataset_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
        print(f"  Device: {device or 'auto'}")
        print(f"  Project: {project}")
        print(f"  Name: {name}")
        print()

        # Initialize ClearML if enabled
        if use_clearml:
            if clearml_project_name and clearml_task_name:
                print(f"Initializing ClearML task: {clearml_project_name}/{clearml_task_name}")
                task = Task.init(
                    project_name=clearml_project_name,
                    task_name=clearml_task_name
                )
                task.connect(train_args)
            else:
                print("Warning: ClearML enabled but project_name or task_name not provided. Skipping ClearML...")


        # Add prediction plot callback if requested
        if num_prediction_plots > 0:
            callback = PredictionPlotCallback(
                num_predictions=num_prediction_plots,
                use_clearml=use_clearml
            )
            model.add_callback("on_val_batch_end", callback)
            print(f"Added prediction plot callback: {num_prediction_plots} plots per epoch")

        # Train the model
        results = model.train(**train_args)

        print("\nFine-tuning completed!")
        print(f"Best model saved at: {Path(project) / name / 'weights' / 'best.pt'}")

        return model

    @staticmethod
    def create_dataset_yaml(
        dataset_dir: str,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: Optional[str] = None,
        num_classes: int = 1,
        class_names: Optional[list] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Create a YOLOv8 dataset YAML file.

        Args:
            dataset_dir: Root directory of the dataset
            train_dir: Training images directory (relative to dataset_dir)
            val_dir: Validation images directory (relative to dataset_dir)
            test_dir: Test images directory (optional, relative to dataset_dir)
            num_classes: Number of classes
            class_names: List of class names (default: ["object"])
            output_path: Output path for YAML file (default: dataset_dir/dataset.yaml)

        Returns:
            Path to created YAML file
        """
        dataset_path = Path(dataset_dir).resolve()

        if class_names is None:
            class_names = ["object"]

        if len(class_names) != num_classes:
            raise ValueError(
                f"Number of class names ({len(class_names)}) must match num_classes ({num_classes})"
            )

        # Prepare paths
        train_path = dataset_path / train_dir
        val_path = dataset_path / val_dir
        test_path = dataset_path / test_dir if test_dir else None

        # Verify directories exist
        if not train_path.exists():
            raise FileNotFoundError(f"Training directory not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_path}")
        if test_path and not test_path.exists():
            raise FileNotFoundError(f"Test directory not found: {test_path}")

        # Create YAML content
        yaml_content = {
            "path": str(dataset_path),
            "train": train_dir,
            "val": val_dir,
        }

        if test_dir:
            yaml_content["test"] = test_dir

        yaml_content["nc"] = num_classes
        yaml_content["names"] = class_names

        # Write YAML file
        if output_path is None:
            output_path = dataset_path / "dataset.yaml"
        else:
            output_path = Path(output_path)

        with open(output_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"Dataset YAML created at: {output_path}")
        return str(output_path)


class PredictionPlotCallback:
    """Callback to plot validation predictions and upload to ClearML."""
    
    def __init__(
        self,
        num_predictions: int = 4,
        use_clearml: bool = False,
    ):
        """
        Initialize the prediction plot callback.
        
        Args:
            num_predictions: Number of prediction plots to generate per epoch
            use_clearml: Whether to upload plots to ClearML
        """
        self.num_predictions = num_predictions
        self.use_clearml = use_clearml
        self.plots_generated = 0
        self.current_epoch = 0
    
    def __call__(self, validator):
        """
        Callback function called after each validation batch.
        
        Args:
            validator: Ultralytics validator object
        """
        # Check if we've generated enough plots for this epoch
        if self.plots_generated >= self.num_predictions:
            return
        
        # Get current epoch from trainer
        frame = inspect.currentframe()
        try:
            # Navigate up the call stack to find trainer
            for _ in range(5):  # Check up to 5 frames up
                frame = frame.f_back
                if frame is None:
                    break
                locals_dict = frame.f_locals
                
                # Check if we're in the trainer context
                if 'self' in locals_dict:
                    trainer_obj = locals_dict['self']
                    if hasattr(trainer_obj, 'epoch'):
                        epoch = trainer_obj.epoch + 1  # 0-indexed to 1-indexed
                        if epoch != self.current_epoch:
                            # New epoch, reset counter
                            self.current_epoch = epoch
                            self.plots_generated = 0
                        break
        except Exception:
            pass
        
        # Randomly decide whether to plot this batch (to limit number of plots)
        if random.random() > (self.num_predictions / 10.0):  # Approximate probability
            return
        
        try:
            # Get batch and predictions from validator context
            frame = inspect.currentframe().f_back.f_back
            if frame is None:
                return
            
            v = frame.f_locals
            
            if 'batch' not in v or 'batch_i' not in v:
                return
            
            batch = v['batch']
            batch_i = v['batch_i']
            preds = v.get('preds', None)
            
            # Plot validation samples
            val_samples_img = validator.plot_val_samples(batch, batch_i)
            
            # Plot predictions if available
            if preds is not None:
                pred_img = validator.plot_predictions(batch, preds, batch_i)
            else:
                pred_img = None
            
            # Upload to ClearML if enabled
            if self.use_clearml:
                try:
                    task = Task.current_task()
                    if task:
                        if val_samples_img is not None:
                            task.get_logger().report_image(
                                title=f"Validation Samples Epoch {self.current_epoch}",
                                series=f"Batch {batch_i}",
                                iteration=self.current_epoch,
                                image=val_samples_img,
                            )
                        
                        if pred_img is not None:
                            task.get_logger().report_image(
                                title=f"Predictions Epoch {self.current_epoch}",
                                series=f"Batch {batch_i}",
                                iteration=self.current_epoch,
                                image=pred_img,
                            )
                except Exception as e:
                    print(f"Warning: Failed to upload plots to ClearML: {e}")
            
            self.plots_generated += 1
            
        except Exception as e:
            print(f"Warning: Failed to generate prediction plots: {e}")