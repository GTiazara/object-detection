"""Training script for fine-tuning YOLO models using configuration file."""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import albumentations as A
import cv2

from src.training.trainer import FineTuner

from dotenv import load_dotenv

load_dotenv()



def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses conf_train.yaml in project root.
        
    Returns:
        Dictionary containing configuration values with defaults.
    """
    project_root = Path(__file__).parent.parent.parent
    
    if config_path is None:
        config_path = project_root / "conf_train.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return config
    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {e}") from e


def build_augmentations(aug_config: Dict[str, Any]) -> list:
    """
    Build albumentations transforms from config.
    
    Args:
        aug_config: Augmentation configuration dictionary
        
    Returns:
        List of albumentations transforms
    """
    if not aug_config.get('enabled', False):
        return []
    
    transforms = []
    transform_map = {
        'Blur': A.Blur,
        'CLAHE': A.CLAHE,
        'RandomBrightnessContrast': A.RandomBrightnessContrast,
        'HueSaturationValue': A.HueSaturationValue,
        'ChannelShuffle': A.ChannelShuffle,
        'ChromaticAberration': A.ChromaticAberration,
        'Dithering': A.Dithering,
    }
    
    for transform_cfg in aug_config.get('transforms', []):
        transform_type = transform_cfg.pop('type')
        if transform_type not in transform_map:
            print(f"Warning: Unknown transform type '{transform_type}', skipping")
            continue
        
        # Handle interpolation string conversion
        if 'interpolation' in transform_cfg:
            interp_str = transform_cfg['interpolation']
            if isinstance(interp_str, str):
                transform_cfg['interpolation'] = getattr(cv2, interp_str, cv2.INTER_LINEAR)
        
        transform_class = transform_map[transform_type]
        transforms.append(transform_class(**transform_cfg))
    
    return transforms


def main():
    """Main training function using configuration file."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO model on custom dataset using configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: conf_train.yaml in project root)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract configuration sections
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    training_cfg = config.get('training', {})
    output_cfg = config.get('output', {})
    aug_cfg = config.get('augmentation', {})
    clearml_cfg = config.get('clearml', {})
    
    # Dataset configuration
    dataset_dir = Path(dataset_cfg.get('dataset_dir'))
    dataset_yaml_path = dataset_cfg.get('dataset_yaml')
    
    if dataset_yaml_path:
        dataset_yaml = Path(dataset_yaml_path)
    else:
        dataset_yaml = dataset_dir / "data.yaml"
    
    # Create dataset YAML if it doesn't exist
    if not dataset_yaml.exists():
        print("Creating dataset YAML...")
        dataset_yaml = Path(FineTuner.create_dataset_yaml(
            dataset_dir=str(dataset_dir),
            train_dir=dataset_cfg.get('train_dir', 'train'),
            val_dir=dataset_cfg.get('val_dir', 'val'),
            test_dir=dataset_cfg.get('test_dir'),
            num_classes=dataset_cfg.get('num_classes', 1),
            class_names=dataset_cfg.get('class_names', ['object']),
        ))
    else:
        print(f"Using existing dataset YAML: {dataset_yaml}")
    
    # Initialize fine-tuner
    base_model_path = model_cfg.get('base_model_path')
    base_model_variant = model_cfg.get('base_model_variant')
    
    # FineTuner requires at least one of base_model_path or base_model_variant
    fine_tuner_kwargs = {}
    if base_model_path:
        fine_tuner_kwargs['base_model_path'] = base_model_path
    if base_model_variant:
        fine_tuner_kwargs['base_model_variant'] = base_model_variant
    
    fine_tuner = FineTuner(**fine_tuner_kwargs)
    
    # Build augmentations
    custom_transforms = build_augmentations(aug_cfg) if aug_cfg.get('enabled', False) else None
    
    # Prepare training arguments
    train_kwargs = {}
    if custom_transforms:
        train_kwargs = {
            'augmentations': custom_transforms,
        }
    else:
        train_kwargs = { 
            'hsv_h': 0.5,
            'hsv_s': 1.0,
            'hsv_v': 0.5,
            'degrees': 180,
            'translate': 0.25,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 0,
        }
    
    # Add optional training parameters
    if 'amp' in training_cfg:
        train_kwargs['amp'] = training_cfg['amp']
    if 'pretrained' in training_cfg:
        train_kwargs['pretrained'] = training_cfg['pretrained']
    if 'cache' in training_cfg:
        train_kwargs['cache'] = training_cfg['cache']
    if 'plots' in training_cfg:
        train_kwargs['plots'] = training_cfg['plots']
    
    # Train the model
    print("\nStarting fine-tuning...")
    model = fine_tuner.train(
        dataset_yaml=str(dataset_yaml),
        epochs=training_cfg.get('epochs', 100),
        imgsz=training_cfg.get('imgsz', 640),
        batch=training_cfg.get('batch', 16),
        device=training_cfg.get('device'),
        project=output_cfg.get('project', 'training/runs'),
        name=output_cfg.get('name', 'fine_tune'),
        patience=training_cfg.get('patience', 50),
        save_period=training_cfg.get('save_period', 10),
        use_clearml=clearml_cfg.get('enabled', False),
        clearml_project_name=clearml_cfg.get('project_name'),
        clearml_task_name=clearml_cfg.get('task_name'),
        **train_kwargs,
    )
    
    print("\n" + "="*60)
    print("Fine-tuning completed!")
    print("="*60)
    output_name = output_cfg.get('name', 'fine_tune')
    output_project = output_cfg.get('project', 'training/runs')
    print(f"\nBest model saved at:")
    print(f"  {output_project}/{output_name}/weights/best.pt")
    print(f"\nYou can now use this model with Detector:")
    print(f"  from src.model_loader import ModelLoader")
    print(f"  loader = ModelLoader()")
    print(f"  model = loader.load_local_model('{output_project}/{output_name}/weights/best.pt')")
    print("="*60)


if __name__ == "__main__":
    main()

