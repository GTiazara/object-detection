"""Command-line training script for fine-tuning YOLOv8 models."""

import argparse
from pathlib import Path
from src.training.trainer import FineTuner


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 model on custom dataset"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        required=True,
        help="Path to dataset YAML file (YOLOv8 format)"
    )
    
    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="training0",
        help="Base model variant to fine-tune (default: training0)"
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Local path to base model (overrides --base-model)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        choices=[640, 960, 1280],
        help="Image size for training (default: 960)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or None for auto)"
    )
    
    # Output arguments
    parser.add_argument(
        "--project",
        type=str,
        default="training/runs",
        help="Project directory for saving results (default: training/runs)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fine_tune",
        help="Experiment name (default: fine_tune)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = FineTuner(
        base_model_variant=args.base_model,
        base_model_path=args.base_model_path,
    )
    
    # Train the model
    model = fine_tuner.train(
        dataset_yaml=args.dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )
    
    print("\nTraining completed successfully!")
    print(f"You can now use the trained model with the same output format as before.")


if __name__ == "__main__":
    main()

