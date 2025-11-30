"""Main detection functionality for airplanes in images."""

from pathlib import Path
from typing import List, Union, Optional, Tuple, Any
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Try to import yolov5
try:
    import yolov5
    YOLOV5_AVAILABLE = True
except ImportError:
    YOLOV5_AVAILABLE = False


class AirplaneDetector:
    """Detects airplanes in satellite imagery using YOLO models."""
    
    def __init__(self, model: Any, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the airplane detector.
        
        Args:
            model: Loaded YOLO model (YOLOv8/YOLOv9 from ultralytics or YOLOv5 from yolov5)
            conf_threshold: Confidence threshold for detections (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.45)
        """
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Update model thresholds if it's YOLOv5
        if YOLOV5_AVAILABLE and hasattr(model, 'conf'):
            model.conf = conf_threshold
            model.iou = iou_threshold
        
        # Detect if it's YOLOv5 model
        # YOLOv5 models from yolov5 library have render() method but not plot() method
        # Ultralytics models have plot() method but not render() method
        self.is_yolov5 = hasattr(model, 'render') and not hasattr(model, 'plot')
    
    def detect(
        self,
        image_path: Union[str, Path],
        imgsz: int = 960,
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Union[Results, Any]:
        """
        Detect airplanes in an image.
        
        Args:
            image_path: Path to the input image
            imgsz: Input image size (default: 960, can use 640 or 1280)
            save: Whether to save the results
            save_dir: Directory to save results (default: same as input image)
            show: Whether to display the results
        
        Returns:
            YOLO Results object containing detections (or YOLOv5 results)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Run inference based on model type
        if self.is_yolov5:
            # YOLOv5 inference
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            results = self.model(str(image_path), size=imgsz)
            
            # Print detection summary
            if results is not None:
                num_detections = len(results.xyxy[0]) if len(results.xyxy) > 0 else 0
                print(f"Detected {num_detections} airplane(s)")
                
                if num_detections > 0:
                    confidences = results.xyxy[0][:, 4].cpu().numpy() if len(results.xyxy[0]) > 0 else []
                    print(f"Confidence scores: {confidences}")
            
            return results
        else:
            # Ultralytics YOLO (YOLOv8/YOLOv9) inference
            results = self.model.predict(
                source=str(image_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=imgsz,
                save=save,
                save_dir=str(save_dir) if save_dir else None,
                show=show,
            )
            
            # Print detection summary
            if results and len(results) > 0:
                result = results[0]
                num_detections = len(result.boxes) if result.boxes is not None else 0
                print(f"Detected {num_detections} airplane(s)")
                
                if num_detections > 0:
                    confidences = result.boxes.conf.cpu().numpy()
                    print(f"Confidence scores: {confidences}")
            
            return results[0] if results else None
    
    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        imgsz: int = 960,
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Results]:
        """
        Detect airplanes in multiple images.
        
        Args:
            image_paths: List of paths to input images
            imgsz: Input image size
            save: Whether to save the results
            save_dir: Directory to save results
        
        Returns:
            List of YOLO Results objects
        """
        results = []
        for image_path in image_paths:
            result = self.detect(
                image_path=image_path,
                imgsz=imgsz,
                save=save,
                save_dir=save_dir,
            )
            results.append(result)
        return results
    
    def visualize(
        self,
        image_path: Union[str, Path],
        results: Any,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Visualize detection results on the image.
        
        Args:
            image_path: Path to the original image
            results: YOLO Results object (YOLOv8/YOLOv9) or YOLOv5 results
            output_path: Optional path to save the visualized image
        
        Returns:
            Annotated image as numpy array
        """
        if self.is_yolov5:
            # YOLOv5 visualization
            annotated_img = results.render()[0]  # Get first image from batch
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR) if len(annotated_img.shape) == 3 else annotated_img
        else:
            # Ultralytics YOLO visualization
            annotated_img = results.plot()
        
        if output_path:
            cv2.imwrite(str(output_path), annotated_img)
            print(f"Visualization saved to {output_path}")
        
        return annotated_img
    
    def get_detections_summary(self, results: Any) -> dict:
        """
        Get a summary of detections.
        
        Args:
            results: YOLO Results object (YOLOv8/YOLOv9) or YOLOv5 results
        
        Returns:
            Dictionary with detection summary
        """
        if self.is_yolov5:
            # YOLOv5 results format
            if results is None or len(results.xyxy) == 0:
                return {
                    "num_detections": 0,
                    "boxes": [],
                    "confidences": [],
                }
            
            boxes_list = results.xyxy[0].cpu().numpy() if len(results.xyxy[0]) > 0 else np.array([])
            if len(boxes_list) == 0:
                return {
                    "num_detections": 0,
                    "boxes": [],
                    "confidences": [],
                }
            
            boxes = boxes_list[:, :4]  # x1, y1, x2, y2
            confidences = boxes_list[:, 4]  # confidence scores
            
            return {
                "num_detections": len(boxes),
                "boxes": boxes.tolist(),
                "confidences": confidences.tolist(),
                "average_confidence": float(np.mean(confidences)) if len(confidences) > 0 else 0.0,
            }
        else:
            # Ultralytics YOLO (YOLOv8/YOLOv9) results format
            if results.boxes is None:
                return {
                    "num_detections": 0,
                    "boxes": [],
                    "confidences": [],
                }
            
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            return {
                "num_detections": len(boxes),
                "boxes": boxes.tolist(),
                "confidences": confidences.tolist(),
                "average_confidence": float(np.mean(confidences)) if len(confidences) > 0 else 0.0,
            }



