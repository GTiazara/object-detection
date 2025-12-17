"""Main detection functionality for object detection in images using YOLO models."""

import gc
from pathlib import Path
from typing import List, Union, Optional, Tuple, Any
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

import rasterio
from rasterio.crs import CRS


class Detector:
    """Detects objects in images using YOLO models."""
    
    def __init__(self, model: Any, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the detector.
        
        Args:
            model: Loaded YOLO model (YOLOv8/YOLOv9 from ultralytics)
            conf_threshold: Confidence threshold for detections (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.45)
        """
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(
        self,
        image_path: Union[str, Path],
        imgsz: int = 960,
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
        **kwargs,
    ) -> Union[Results, Any]:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to the input image
            imgsz: Input image size (default: 960, can use 640 or 1280)
            save: Whether to save the results
            save_dir: Directory to save results (default: same as input image)
            show: Whether to display the results
        
        Returns:
            YOLO Results object containing detections
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Ultralytics YOLO (YOLOv8/YOLOv9) inference
        predict_args={
            "source": str(image_path),
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": imgsz,
            "save": save,
            "project": str(save_dir) if save_dir else None,
            "show": show,
            **kwargs,
        }
        results = self.model.predict(**predict_args)
        
        # Print detection summary
        if results and len(results) > 0:
            result = results[0]
            num_detections = len(result.boxes) if result.boxes is not None else 0
            print(f"Detected {num_detections} object(s)")
            
            if num_detections > 0:
                # Move to CPU immediately to free GPU memory
                confidences = result.boxes.conf.cpu().numpy()
                print(f"Confidence scores: {confidences}")
        
        # Return first result
        # Note: Tensors will be moved to CPU when converting to numpy in get_detections_summary()
        # This avoids modifying read-only properties while still freeing GPU memory during conversion
        return results[0] if results and len(results) > 0 else None
    
    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        imgsz: int = 960,
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Results]:
        """
        Detect objects in multiple images.
        
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
            results: YOLO Results object (YOLOv8/YOLOv9)
            output_path: Optional path to save the visualized image
        
        Returns:
            Annotated image as numpy array
        """
        # Ultralytics YOLO visualization
        # Note: results.plot() handles GPU tensors internally, so we don't need to modify them
        annotated_img = results.plot()
        
        if output_path:
            output_path = Path(output_path)
            image_path = Path(image_path)
            
            # Check if original image is a georeferenced TIF and preserve georeferencing
            is_geotiff = (
                image_path.suffix.lower() in ['.tif', '.tiff'] and
                image_path.exists()
            )
            
            if is_geotiff:
                try:
                    # Read georeferencing from original image
                    with rasterio.open(image_path) as src:
                        transform = src.transform
                        original_height = src.height
                        original_width = src.width
                        nodata = src.nodata
                        tags = src.tags().copy()
                        meta = src.meta.copy()
                        colorinterp = getattr(src, 'colorinterp', None)
                        
                        if transform.is_identity:
                            raise ValueError("Image does not have a valid georeferencing transform")
                    
                    # Set CRS to EPSG:4326
                    crs = CRS.from_epsg(4326)
                    
                    # Get annotated image dimensions and resize if needed
                    if len(annotated_img.shape) == 3:
                        annotated_height, annotated_width = annotated_img.shape[:2]
                    else:
                        annotated_height, annotated_width = annotated_img.shape
                    
                    if annotated_height != original_height or annotated_width != original_width:
                        annotated_img = cv2.resize(
                            annotated_img, 
                            (original_width, original_height), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    # Convert BGR to RGB for rasterio (create a copy for processing)
                    if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    else:
                        annotated_img_rgb = annotated_img.copy()
                    
                    # Prepare for rasterio: (channels, height, width)
                    if len(annotated_img_rgb.shape) == 2:
                        annotated_img_rgb = np.expand_dims(annotated_img_rgb, axis=2)
                    if len(annotated_img_rgb.shape) == 3:
                        annotated_img_rgb = np.transpose(annotated_img_rgb, (2, 0, 1))
                    
                    # Build output profile
                    output_profile = {
                        'driver': 'GTiff',
                        'height': original_height,
                        'width': original_width,
                        'count': annotated_img_rgb.shape[0],
                        'dtype': annotated_img_rgb.dtype,
                        'crs': crs,
                        'transform': transform,
                        'compress': 'lzw',
                        'tiled': True,
                    }
                    
                    if nodata is not None:
                        output_profile['nodata'] = nodata
                    
                    # Preserve metadata from original
                    for key in ['photometric', 'interleave', 'blockxsize', 'blockysize']:
                        if key in meta:
                            output_profile[key] = meta[key]
                    
                    # Write GeoTIFF
                    with rasterio.open(output_path, 'w', **output_profile) as dst:
                        dst.write(annotated_img_rgb)
                        dst.crs = crs
                        dst.transform = transform
                        if tags:
                            dst.update_tags(**tags)
                        if colorinterp is not None:
                            try:
                                dst.colorinterp = colorinterp
                            except Exception:
                                pass
                    
                    print(f"Visualization saved to {output_path} (georeferenced, CRS: EPSG:4326)")
                    # Clear large arrays after writing to free memory immediately
                    del annotated_img_rgb
                    # Also clear the original annotated image reference
                    del annotated_img
                    # Force garbage collection for large images to prevent memory accumulation
                    import gc
                    gc.collect()
                except Exception as e:
                    print(f"Warning: Could not preserve georeferencing: {e}")
                    cv2.imwrite(str(output_path), annotated_img)
                    print(f"Visualization saved to {output_path} (non-georeferenced)")
            else:
                # Regular image save (non-georeferenced)
                cv2.imwrite(str(output_path), annotated_img)
                print(f"Visualization saved to {output_path}")
        
        # Return annotated image (caller should clean up if not needed)
        # Note: For large images, caller should explicitly delete this to free memory
        return annotated_img
    
    def get_detections_summary(self, results: Any) -> dict:
        """
        Get a summary of detections.
        
        Args:
            results: YOLO Results object (YOLOv8/YOLOv9)
        
        Returns:
            Dictionary with detection summary
        """
        # Ultralytics YOLO (YOLOv8/YOLOv9) results format
        if results.boxes is None:
            return {
                "num_detections": 0,
                "boxes": [],
                "confidences": [],
            }
        
        # Ensure tensors are on CPU before converting to numpy
        # Use .clone() to avoid keeping references to GPU tensors
        boxes = results.boxes.xyxy.cpu().clone().numpy()
        confidences = results.boxes.conf.cpu().clone().numpy()
        
        # Convert to lists immediately to release numpy arrays
        boxes_list = boxes.tolist()
        confidences_list = confidences.tolist()
        
        # Clear intermediate arrays immediately
        del boxes, confidences
        
        return {
            "num_detections": len(boxes_list),
            "boxes": boxes_list,
            "confidences": confidences_list,
            "average_confidence": float(np.mean(confidences_list)) if len(confidences_list) > 0 else 0.0,
        }

