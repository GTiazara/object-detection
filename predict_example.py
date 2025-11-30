"""Example script for airplane detection using Ultralytics YOLO.

This example demonstrates how to use the model loader and detector
following the Ultralytics documentation pattern:
https://docs.ultralytics.com/fr/usage/python/#predict
"""

from pathlib import Path
from src.model_loader import ModelLoader
from src.detector import Detector

# Example: Detect airplanes in a single image
def example_single_image():
    """Example of detecting airplanes in a single image."""
    print("=" * 60)
    print("Example: Single Image Detection")
    print("=" * 60)
    
    # Load model (will download from Hugging Face if not present)
    loader = ModelLoader()
    model = loader.load_model(
        # model_variant="training0",  # or "training", "transfer", etc.
        local_path="C:/Users/tiaza/Documents/perso/personal_project/object-detection/training/runs/my_airplane_detection5/weights/best.pt"  # Uncomment to use local model
    )
    
    # Initialize detector
    detector = Detector(model, conf_threshold=0.25)
    
    # Detect airplanes (using TIF file from input directory)
    # Find first TIF file in input directory
    input_dir = Path("C:/Users/tiaza/Documents/perso/personal_project/object-detection/data/input_test")
    tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.TIF"))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        print("Please place your TIF image in data/input/ directory")
        return
    
    image_path = tif_files[0]  # Use first TIF file found
    print(f"Using image: {image_path}")
    
    if image_path.exists():
        # Use the detector's detect method (which uses model.predict internally)
        result = detector.detect(
            image_path=image_path,
            imgsz=640,  # Can use 640, 960, or 1280
            save=True,
            save_dir="data/output",
        )
        
        # Get summary
        summary = detector.get_detections_summary(result)
        print(f"\nDetection Summary:")
        print(f"  Number of objects: {summary['num_detections']}")
        print(f"  Average confidence: {summary['average_confidence']:.3f}")
        
        # Visualize
        output_path = Path("data/output") / f"{image_path.stem}_detections.jpg"
        detector.visualize(image_path, result, output_path)
        
        # Alternative: Use model.predict directly (as per Ultralytics docs)
        # results = model.predict(
        #     source=str(image_path),
        #     conf=0.25,
        #     imgsz=960,
        #     save=True,
        #     save_dir="data/output",
        # )
    else:
        print(f"Image not found: {image_path}")
        print("Please place your image in data/input/ directory")


if __name__ == "__main__":
    # Run YOLO example
    example_single_image()



