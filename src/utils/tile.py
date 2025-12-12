"""Module for tiling large images and merging detection results.

This module provides functionality to split large images into tiles,
run detection on each tile, and merge the results back together.
It can be used with both H5 files and regular image files.
"""

from typing import List, Tuple, Any, Optional
import numpy as np
import cv2
import torch


def split_image_into_tiles(
    image: np.ndarray,
    tile_size: int,
    overlap: int = 0
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split an image into tiles with optional overlap.
    
    Args:
        image: Input image array (H, W, C)
        tile_size: Size of each tile (square tiles)
        overlap: Overlap between tiles in pixels
        
    Returns:
        List of tuples (tile_image, (y_offset, x_offset)) where offsets are in original image coordinates
    """
    h, w = image.shape[:2]
    tiles = []
    stride = tile_size - overlap
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Calculate tile boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Extract tile
            tile = image[y:y_end, x:x_end]
            
            # Pad tile if it's smaller than tile_size (at image boundaries)
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            tiles.append((tile, (y, x)))
            x += stride
        
        y += stride
    
    return tiles


def merge_tile_detections(
    all_detections: List[Tuple[Any, Tuple[int, int]]],
    original_shape: Tuple[int, int],
    merge_nms_iou: float = 0.45
) -> Any:
    """
    Merge detection results from multiple tiles into a single result.
    
    This function handles both bounding boxes and masks (for instance segmentation).
    
    Args:
        all_detections: List of tuples (detection_result, (y_offset, x_offset))
        original_shape: Original image shape (height, width)
        merge_nms_iou: IoU threshold for NMS when merging
        
    Returns:
        Merged detection result (mock result object with boxes and masks)
    """
    all_boxes = []
    all_confidences = []
    all_classes = []
    all_masks = []
    original_height, original_width = original_shape
    
    for result, (y_offset, x_offset) in all_detections:
        if result is None:
            continue
        
        # Extract boxes and confidences
        boxes = None
        confidences = None
        classes = None
        masks = None
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else np.zeros(len(boxes))
            
            # Extract masks if available (for instance segmentation)
            if hasattr(result, 'masks') and result.masks is not None:
                try:
                    # Masks are typically stored as tensor of shape (N, H, W) or (N, 1, H, W)
                    masks_data = result.masks.data
                    if masks_data is not None:
                        masks = masks_data.cpu().numpy()
                        # Handle different mask formats
                        if len(masks.shape) == 4 and masks.shape[1] == 1:
                            masks = masks[:, 0, :, :]  # Remove channel dimension
                        elif len(masks.shape) == 3:
                            # Already in (N, H, W) format
                            pass
                        else:
                            masks = None
                except Exception as e:
                    masks = None
        elif hasattr(result, 'xyxy') and len(result.xyxy) > 0:
            boxes = result.xyxy[0].cpu().numpy() if len(result.xyxy[0]) > 0 else np.array([])
            if len(boxes) > 0:
                confidences = boxes[:, 4] if boxes.shape[1] > 4 else np.ones(len(boxes))
                boxes = boxes[:, :4]
                classes = np.zeros(len(boxes))
            else:
                continue
        else:
            continue
        
        if boxes is None or len(boxes) == 0:
            continue
        
        # Adjust box coordinates to original image coordinates
        boxes[:, 0] += x_offset  # x1
        boxes[:, 1] += y_offset  # y1
        boxes[:, 2] += x_offset  # x2
        boxes[:, 3] += y_offset  # y2
        
        # Adjust masks to original image coordinates
        num_detections = len(boxes)
        adjusted_masks = []
        
        if masks is not None and len(masks) > 0:
            # Process each mask
            for i in range(num_detections):
                if i < len(masks):
                    mask = masks[i]
                    
                    # Get the box coordinates (already adjusted to original image)
                    box = boxes[i]
                    x1, y1, x2, y2 = box[:4]
                    box_w = max(1, int(x2 - x1))
                    box_h = max(1, int(y2 - y1))
                    
                    # Resize mask to match box size
                    # Masks from YOLO are typically in model output size (e.g., 640x640)
                    # We need to resize them to the actual box size
                    if mask.shape[0] != box_h or mask.shape[1] != box_w:
                        mask_resized = cv2.resize(
                            mask.astype(np.float32), 
                            (box_w, box_h), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        mask_resized = mask.astype(np.float32)
                    
                    # Create a full-size mask for original image
                    full_mask = np.zeros((original_height, original_width), dtype=np.float32)
                    
                    # Calculate position in original image (clamp to image bounds)
                    x1_int = max(0, min(int(x1), original_width - 1))
                    y1_int = max(0, min(int(y1), original_height - 1))
                    x2_int = max(0, min(int(x2), original_width))
                    y2_int = max(0, min(int(y2), original_height))
                    
                    # Ensure we have valid dimensions
                    mask_h = y2_int - y1_int
                    mask_w = x2_int - x1_int
                    
                    if mask_h > 0 and mask_w > 0:
                        # Resize mask to fit exactly in the box region
                        if mask_resized.shape[0] != mask_h or mask_resized.shape[1] != mask_w:
                            mask_resized = cv2.resize(
                                mask_resized,
                                (mask_w, mask_h),
                                interpolation=cv2.INTER_LINEAR
                            )
                        # Place mask in correct position
                        full_mask[y1_int:y2_int, x1_int:x2_int] = mask_resized
                    
                    adjusted_masks.append(full_mask)
                else:
                    # Create empty mask if no mask available for this detection
                    adjusted_masks.append(np.zeros((original_height, original_width), dtype=np.float32))
        else:
            # Add empty masks for detections without masks
            for _ in range(num_detections):
                adjusted_masks.append(np.zeros((original_height, original_width), dtype=np.float32))
        
        all_masks.append(adjusted_masks)
        
        all_boxes.append(boxes)
        all_confidences.append(confidences)
        all_classes.append(classes)
    
    if not all_boxes:
        # Return empty result
        class EmptyResult:
            boxes = None
            masks = None
        return EmptyResult()
    
    # Concatenate all detections
    merged_boxes = np.concatenate(all_boxes, axis=0)
    merged_confidences = np.concatenate(all_confidences, axis=0)
    merged_classes = np.concatenate(all_classes, axis=0)
    
    # Concatenate all masks
    merged_masks_list = []
    for masks in all_masks:
        merged_masks_list.extend(masks)
    
    # Apply NMS to remove duplicates
    nms_indices = None
    if len(merged_boxes) > 0:
        # Use OpenCV's NMS
        nms_result = cv2.dnn.NMSBoxes(
            merged_boxes.tolist(),
            merged_confidences.tolist(),
            score_threshold=0.0,
            nms_threshold=merge_nms_iou
        )
        
        if nms_result is not None and len(nms_result) > 0:
            nms_indices = nms_result.flatten()
            merged_boxes = merged_boxes[nms_indices]
            merged_confidences = merged_confidences[nms_indices]
            merged_classes = merged_classes[nms_indices]
            
            # Filter masks to match NMS indices
            if merged_masks_list:
                merged_masks_list = [merged_masks_list[i] for i in nms_indices]
    
    # Convert masks list to array if we have any masks
    merged_masks = None
    if merged_masks_list and len(merged_masks_list) > 0:
        # Ensure we have the same number of masks as boxes
        if len(merged_masks_list) == len(merged_boxes):
            merged_masks = np.array(merged_masks_list)
        else:
            # Pad or trim masks to match box count
            if len(merged_masks_list) < len(merged_boxes):
                # Add empty masks for missing ones
                for _ in range(len(merged_boxes) - len(merged_masks_list)):
                    merged_masks_list.append(np.zeros((original_height, original_width), dtype=np.float32))
            else:
                # Trim excess masks
                merged_masks_list = merged_masks_list[:len(merged_boxes)]
            merged_masks = np.array(merged_masks_list)
    
    # Create a mock result object compatible with detector
    class MergedResult:
        def __init__(self, boxes, confidences, classes, masks=None):
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            conf_tensor = torch.tensor(confidences, dtype=torch.float32)
            cls_tensor = torch.tensor(classes, dtype=torch.float32) if len(classes) > 0 else None
            
            self.boxes = type('Boxes', (), {
                'xyxy': boxes_tensor,
                'conf': conf_tensor,
                'cls': cls_tensor
            })()
            
            # Add masks if available
            if masks is not None:
                # Convert to tensor format expected by YOLO
                masks_tensor = torch.tensor(masks, dtype=torch.float32)
                if len(masks_tensor.shape) == 2:
                    masks_tensor = masks_tensor.unsqueeze(0)
                self.masks = type('Masks', (), {
                    'data': masks_tensor
                })()
            else:
                self.masks = None
            
            # Add plot method for visualization
            self.plot = lambda: None  # Will be handled by detector.visualize
    
    return MergedResult(merged_boxes, merged_confidences, merged_classes, merged_masks)


class TileProcessor:
    """
    Class for processing large images by splitting them into tiles.
    
    This class can be used with both H5 files and regular image files.
    
    Example usage with regular image files:
        from src.detector import Detector
        from src.model_loader import ModelLoader
        from src.predict.tile import TileProcessor
        import cv2
        
        # Load model and create detector
        loader = ModelLoader()
        model = loader.load_model(local_path="model.pt")
        detector = Detector(model, conf_threshold=0.25)
        
        # Create tile processor
        tile_processor = TileProcessor(
            detector=detector,
            tile_size=640,
            overlap=50,
            merge_nms_iou=0.45,
            verbose=True
        )
        
        # Load and process image
        image = cv2.imread("large_image.jpg")
        result = tile_processor.process_image(image, imgsz=640)
        
        # Use result for visualization or further processing
        if result and result.boxes is not None:
            summary = detector.get_detections_summary(result)
            print(f"Detected {summary['num_detections']} objects")
    """
    
    def __init__(
        self,
        detector: Any,
        tile_size: int,
        overlap: int = 0,
        merge_nms_iou: float = 0.45,
        verbose: bool = False
    ):
        """
        Initialize the tile processor.
        
        Args:
            detector: Detector instance for running predictions
            tile_size: Size of each tile (square tiles)
            overlap: Overlap between tiles in pixels
            merge_nms_iou: IoU threshold for NMS when merging
            verbose: Print detailed progress information
        """
        self.detector = detector
        self.tile_size = tile_size
        self.overlap = overlap
        self.merge_nms_iou = merge_nms_iou
        self.verbose = verbose
    
    def process_image(
        self,
        image: np.ndarray,
        imgsz: int = 640
    ) -> Any:
        """
        Process a large image by splitting into tiles, running detection, and merging results.
        
        Args:
            image: Input image array (H, W, C)
            imgsz: Input image size for model inference
            
        Returns:
            Merged detection result
        """
        h, w = image.shape[:2]
        
        # Check if tiling is needed
        if h <= self.tile_size and w <= self.tile_size:
            # Image is small enough, process directly
            import tempfile
            from pathlib import Path
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            
            try:
                cv2.imwrite(str(tmp_path), image)
                result = self.detector.detect(
                    image_path=tmp_path,
                    imgsz=imgsz,
                    save=False,
                )
                return result
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        
        # Split image into tiles
        if self.verbose:
            print(f"  Using tiling: image {h}x{w} split into {self.tile_size}x{self.tile_size} tiles (overlap: {self.overlap}px)")
        
        tiles = split_image_into_tiles(image, self.tile_size, self.overlap)
        
        if self.verbose:
            print(f"  Created {len(tiles)} tile(s)")
        
        all_tile_results = []
        import tempfile
        from pathlib import Path
        
        for tile_idx, (tile, (y_offset, x_offset)) in enumerate(tiles, 1):
            if self.verbose:
                print(f"    Processing tile {tile_idx}/{len(tiles)} (offset: {y_offset}, {x_offset})")
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            
            try:
                # Save tile temporarily
                cv2.imwrite(str(tmp_path), tile)
                
                # Run detection on tile
                tile_result = self.detector.detect(
                    image_path=tmp_path,
                    imgsz=imgsz,
                    save=False,
                    # save_dir="data/output",
                )
                
                if tile_result:
                    all_tile_results.append((tile_result, (y_offset, x_offset)))
            finally:
                # Clean up temporary file
                if tmp_path.exists():
                    tmp_path.unlink()
        
        # Merge all tile detections
        if all_tile_results:
            if self.verbose:
                print(f"  Merging {len(all_tile_results)} tile detection(s)")
            result = merge_tile_detections(all_tile_results, (h, w), self.merge_nms_iou)
            return result
        else:
            # Return empty result
            class EmptyResult:
                boxes = None
                masks = None
            return EmptyResult()

