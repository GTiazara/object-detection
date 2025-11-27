"""Example script for airplane detection using Ultralytics YOLO or MMDetection.

This example demonstrates how to use the model loader and detector
following the Ultralytics documentation pattern:
https://docs.ultralytics.com/fr/usage/python/#predict

For MMDetection models, see:
https://github.com/open-mmlab/mmdetection
"""

from pathlib import Path
from airplane_detection.model_loader import ModelLoader
from airplane_detection.mmdetection_loader import MMDetectionLoader
from airplane_detection.detector import AirplaneDetector

# Example: Detect airplanes in a single image
def example_single_image():
    """Example of detecting airplanes in a single image."""
    print("=" * 60)
    print("Example: Single Image Detection")
    print("=" * 60)
    
    # Load model (will download from Hugging Face if not present)
    # Option 1: Load YOLO model (Ultralytics) - recommended
    loader = ModelLoader()
    model = loader.load_model(
        model_variant="training",  # or "training", "transfer", etc.
        # local_path="path/to/your/model.pt"  # Uncomment to use local model
    )
    
    # Option 2: Load MMDetection model (uncomment to use)
    # Note: Requires MMDetection installed: pip install mmdet mmcv mmengine
    # mmdet_loader = MMDetectionLoader()
    # model = mmdet_loader.load_model(
    #     model_variant="faster_rcnn_r50",  # or "rtmdet_s", "yolox_s"
    #     class_filter=[4],  # COCO class 4 = airplane (None = all classes)
    #     device="cuda:0",  # or "cpu"
    # )
    
    # Initialize detector
    detector = AirplaneDetector(model, conf_threshold=0.25)
    
    # Detect airplanes (using TIF file from input directory)
    # Find first TIF file in input directory
    input_dir = Path("/home/GTiazara/Documents/workspace/get_experience_project/geo-dataset-builder/output")
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
            imgsz=960,  # Can use 640, 960, or 1280
            save=True,
            save_dir="data/output",
        )
        
        # Get summary
        summary = detector.get_detections_summary(result)
        print(f"\nDetection Summary:")
        print(f"  Number of airplanes: {summary['num_detections']}")
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


# Example: Detect airplanes using MMDetection
def example_mmdetection():
    """Example of detecting airplanes using MMDetection models."""
    print("=" * 60)
    print("Example: MMDetection Model Detection")
    print("=" * 60)
    
    # Check if MMDetection is available
    try:
        from airplane_detection.mmdetection_loader import MMDetectionLoader, MMDETECTION_AVAILABLE
        if not MMDETECTION_AVAILABLE:
            print("MMDetection is not installed.")
            print("Install it with: pip install mmdet mmcv mmengine")
            return
    except ImportError:
        print("MMDetection loader not available.")
        return
    
    # Load MMDetection model
    # Available model variants:
    # - "rtmdet_tiny": RTMDet-Tiny (lightweight, fast - recommended)
    # - "rtmdet_s": RTMDet-S (modern, efficient detector)
    # - "faster_rcnn_r50": Faster R-CNN with ResNet50 (accurate but slower)
    # - "yolox_s": YOLOX-S (YOLO variant in MMDetection)
    mmdet_loader = MMDetectionLoader()
    
    # Try to use GPU, fallback to CPU if not available
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model with airplane class filtering (COCO class 4 = airplane)
    # This follows the MMDetection pattern:
    #   from mmdet.apis import init_detector, inference_detector
    #   model = init_detector(config_file, checkpoint_file, device='cpu')
    #   inference_detector(model, 'image.jpg')
    model = mmdet_loader.load_model(
        model_variant="rtmdet_tiny",  # Options: "rtmdet_tiny", "rtmdet_s", "faster_rcnn_r50", "yolox_s"
        class_filter=[4],  # COCO class 4 = airplane (None = all classes)
        device=device,
    )
    
    # Initialize detector
    detector = AirplaneDetector(model, conf_threshold=0.25)
    
    # Detect airplanes (using TIF file from input directory)
    input_dir = Path("/home/GTiazara/Documents/workspace/get_experience_project/geo-dataset-builder/output")
    tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.TIF"))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        print("Please place your TIF image in the directory")
        return
    
    image_path = tif_files[0]  # Use first TIF file found
    print(f"Using image: {image_path}")
    
    if image_path.exists():
        # Use the detector's detect method
        result = detector.detect(
            image_path=image_path,
            imgsz=960,  # Can use 640, 960, or 1280
            save=True,
            save_dir="data/output",
        )
        
        # Get summary
        summary = detector.get_detections_summary(result)
        print(f"\nDetection Summary:")
        print(f"  Number of airplanes: {summary['num_detections']}")
        print(f"  Average confidence: {summary['average_confidence']:.3f}")
        
        # Visualize
        output_path = Path("data/output") / f"{image_path.stem}_mmdet_detections.jpg"
        detector.visualize(image_path, result, output_path)
        
        # Alternative: Use model.predict directly
        # results = model.predict(
        #     source=str(image_path),
        #     conf=0.25,
        #     imgsz=960,
        # )
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    # Run YOLO example (default)
    example_single_image()
    
    # Uncomment to run MMDetection example instead:
    # example_mmdetection()



