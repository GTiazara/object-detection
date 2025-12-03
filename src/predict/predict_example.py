"""Example script for airplane detection using Ultralytics YOLO.

This example demonstrates how to use the model loader and detector
following the Ultralytics documentation pattern:
https://docs.ultralytics.com/fr/usage/python/#predict
"""

from pathlib import Path
from src.model_loader import ModelLoader
from src.detector import Detector

# Example: Detect airplanes in all images in input folder
def example_single_image():
    """Example of detecting airplanes in all images in the input folder."""
    print("=" * 60)
    print("Example: Batch Image Detection")
    print("=" * 60)
    
    # Load model (will download from Hugging Face if not present)
    loader = ModelLoader()
    model = loader.load_model(
        # model_variant="training0",  # or "training", "transfer", etc.
        local_path="/home/GTiazara/Documents/mnt/partage/jzlou/myhddhome/src/object-detection/log/yolo5/weights/best.pt"  # Uncomment to use local model
    )
    
    # Initialize detector
    detector = Detector(model, conf_threshold=0.25)
    
    # Detect airplanes (using all image files from input directory)
    input_dir = Path("/home/GTiazara/Documents/workspace/get_experience_project/geo-dataset-builder/output")
    
    # Supported image formats
    image_extensions = ["*.tif", "*.TIF", "*.tiff", "*.TIFF", 
                       "*.png", "*.PNG", 
                       "*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
                       "*.bmp", "*.BMP",
                       "*.webp", "*.WEBP"]
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
    
    # Remove duplicates (in case of case-insensitive filesystem)
    image_files = list(set(image_files))
    image_files.sort()  # Sort for consistent processing order
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        print("Please place your images (TIF, PNG, JPG, etc.) in data/input_test/ directory")
        return
    
    print(f"Found {len(image_files)} image file(s) to process\n")
    
    # Process each file
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 60)
        
        if image_path.exists():
            # Use the detector's detect method (which uses model.predict internally)
            result = detector.detect(
                image_path=image_path,
                imgsz=640,  # Can use 640, 960, or 1280
                save=False,
                save_dir="data/output",
                augment=True,
                visualize=False,
            )
            
            # Get summary
            summary = detector.get_detections_summary(result)
            print(f"Detection Summary:")
            print(f"  Number of objects: {summary['num_detections']}")
            print(f"  Average confidence: {summary['average_confidence']:.3f}")
            
            # Visualize
            # Use .tif extension for georeferenced TIF images, .jpg for others
            if image_path.suffix.lower() in ['.tif', '.tiff']:
                output_path = Path("data/output") / f"{image_path.stem}_detections.tif"
            else:
                output_path = Path("data/output") / f"{image_path.stem}_detections.jpg"
            detector.visualize(image_path, result, output_path)
            print(f"  Output saved to: {output_path}")
        else:
            print(f"Image not found: {image_path}")
    
    print(f"\n{'=' * 60}")
    print(f"Processing complete! Processed {len(image_files)} file(s).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Run YOLO example
    example_single_image()



