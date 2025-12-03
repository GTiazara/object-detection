"""Main script for object detection in images using YOLO models."""

import argparse
from pathlib import Path
from src.model_loader import ModelLoader
from src.detector import Detector


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Detect objects in images using YOLO models"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input image or directory containing images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov9e"],
        help="Model to use for detection (default: yolov8x)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        choices=[640, 960, 1280],
        help="Input image size (default: 960)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: input directory)",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Path to local model file (overrides --model)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of model",
    )
    
    args = parser.parse_args()
    
    # Initialize model loader
    loader = ModelLoader()
    
    # Load model
    if args.local_model:
        model = loader.load_local_model(args.local_model)
    else:
        model = loader.load_model(args.model, force_download=args.force_download)
    
    # Initialize detector
    detector = Detector(
        model=model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent / "detection_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single image
        print(f"\n{'='*60}")
        print("Processing single image")
        print(f"{'='*60}\n")
        
        result = detector.detect(
            image_path=input_path,
            imgsz=args.imgsz,
            save=True,
            save_dir=output_dir,
        )
        
        if result:
            summary = detector.get_detections_summary(result)
            print(f"\n{'='*60}")
            print("Detection Summary:")
            print(f"{'='*60}")
            print(f"Number of objects detected: {summary['num_detections']}")
            if summary['num_detections'] > 0:
                print(f"Average confidence: {summary['average_confidence']:.3f}")
                print(f"Confidence scores: {[f'{c:.3f}' for c in summary['confidences']]}")
            
            # Save visualization
            vis_path = output_dir / f"{input_path.stem}_detections.jpg"
            detector.visualize(input_path, result, vis_path)
    
    elif input_path.is_dir():
        # Directory of images
        print(f"\n{'='*60}")
        print("Processing directory of images")
        print(f"{'='*60}\n")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        image_paths = [
            p for p in input_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {input_path}")
            return
        
        print(f"Found {len(image_paths)} image(s) to process\n")
        
        results = detector.detect_batch(
            image_paths=image_paths,
            imgsz=args.imgsz,
            save=True,
            save_dir=output_dir,
        )
        
        # Print summary
        total_detections = 0
        for image_path, result in zip(image_paths, results):
            if result:
                summary = detector.get_detections_summary(result)
                num_det = summary['num_detections']
                total_detections += num_det
                print(f"{image_path.name}: {num_det} object(s) detected")
        
        print(f"\n{'='*60}")
        print(f"Total objects detected across all images: {total_detections}")
        print(f"{'='*60}")
    
    else:
        print(f"Error: Input path does not exist: {input_path}")
        return


if __name__ == "__main__":
    main()

