"""Example script for fine-tuning the training0 model on a custom dataset."""

from pathlib import Path
from src.training.trainer import FineTuner
import albumentations as A
import cv2
def main():
    """Example fine-tuning workflow."""
    
    # Path to your dataset
    dataset_dir = "C:/Users/tiaza/Downloads/all_pure_airplane_data"
    
    # Step 1: Create dataset YAML (if not already created)
    dataset_yaml = Path(dataset_dir) / "data.yaml"
    
    if not dataset_yaml.exists():
        print("Creating dataset YAML...")
        FineTuner.create_dataset_yaml(
            dataset_dir=dataset_dir,
            train_dir="train",
            val_dir="val",
            test_dir="test",  # Optional
            num_classes=1,
            class_names=["airplane"],
        )
        dataset_yaml = Path(dataset_dir) / "dataset.yaml"
    else:
        print(f"Using existing dataset YAML: {dataset_yaml}")
    
    # Step 2: Initialize fine-tuner
    # Option A: Use pre-trained model from HuggingFace
    fine_tuner = FineTuner(
        base_model_variant="training0",  # Uses iturslab/Efficient-YOLO-RS-Airplane-Detection
    )
    
    # Option B: Use local model file
    # fine_tuner = FineTuner(
    #     base_model_path="path/to/local/model.pt",
    # )
    
    # Step 3: Train the model
    print("\nStarting fine-tuning...")

    custom_transforms = [
    A.Blur(blur_limit=7, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
     A.ChannelShuffle(p=0.3),
    A.ChromaticAberration(primary_distortion_limit=0.05,secondary_distortion_limit=0.1,mode='green_purple',interpolation=cv2.INTER_LINEAR,p=0.3),
    A.Dithering(method="error_diffusion",n_colors=2, error_diffusion_algorithm="floyd_steinberg", color_mode="grayscale", p=0.3)]


    model = fine_tuner.train(
        dataset_yaml=str(dataset_yaml),
        epochs=100,
        imgsz=640,
        batch=3,
        device="cuda",  # Use "cpu" if no GPU available
        project="training/runs",
        name="my_airplane_detection",
        patience=10,  # Early stopping patience
        save_period=-1,
        plots=True,
        cache= True,
        pretrained=True,
        augmentations=custom_transforms,

    )
    
    print("\n" + "="*60)
    print("Fine-tuning completed!")
    print("="*60)
    print(f"\nBest model saved at:")
    print(f"  training/runs/my_airplane_detection/weights/best.pt")
    print(f"\nYou can now use this model with Detector:")
    print(f"  from model_loader import ModelLoader")
    print(f"  loader = ModelLoader()")
    print(f"  model = loader.load_local_model('training/runs/my_airplane_detection/weights/best.pt')")
    print("="*60)


if __name__ == "__main__":
    main()

