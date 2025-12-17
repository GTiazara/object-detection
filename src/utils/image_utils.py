"""Image processing utilities.

This module provides utility functions for image processing and conversion.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS


def array_to_image(array: np.ndarray) -> np.ndarray:
    """
    Convert array to image format suitable for YOLO prediction.
    
    Args:
        array: Input array (can be 2D or 3D, in either (H, W) or (H, W, C) or (C, H, W) or (C, W, H) format)
        
    Returns:
        Image array with exactly 3 dimensions (height, width, channels)
    """
    # Remove any singleton dimensions first
    array = np.squeeze(array)
    
    # Detect array format and convert to (height, width, channels)
    if array.ndim == 3:
        # Check if first dimension is channels (smaller than the other two)
        if array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
            # Could be (C, H, W) or (C, W, H)
            # If shape[1] > shape[2], could be (C, H, W) with H > W, or (C, W, H) with W > H
            # If shape[1] < shape[2], could be (C, H, W) with H < W, or (C, W, H) with W < H
            # For (C, W, H) -> (H, W, C): transpose (2, 1, 0)
            # For (C, H, W) -> (H, W, C): transpose (1, 2, 0)
            # Since user specified (C, W, H) format, we'll use (2, 1, 0) transpose
            # This converts (channel, width, height) to (height, width, channel)
            array = np.transpose(array, (2, 1, 0))  # (C, W, H) -> (H, W, C)
    
    # Handle different array shapes (now in (H, W) or (H, W, C) format)
    if array.ndim == 2:
        # Grayscale: convert to 3-channel
        image = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3:
        num_channels = array.shape[2]
        if num_channels == 1:
            # Single channel: convert to 3-channel
            image = cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            # RGB: keep as is (no BGR conversion)
            image = array
        elif num_channels == 4:
            # RGBA: use first 3 channels (RGB)
            image = array[:, :, -3:]
        else:
            # More than 4 channels: keep the last 3 channels
            # Take channels [-3, -2, -1] which are the last 3
            image = array[:, :, -3:]
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}. After squeezing: {np.squeeze(array).shape}")
    
    # Ensure exactly 3 dimensions (height, width, channels)
    if image.ndim != 3:
        if image.ndim == 2:
            # Add channel dimension
            image = np.expand_dims(image, axis=2)
        elif image.ndim > 3:
            # Remove extra dimensions
            image = np.squeeze(image)
            if image.ndim != 3:
                raise ValueError(f"Cannot convert array to 3D image. Shape after processing: {image.shape}")
    
    # Ensure exactly 3 channels
    if image.shape[2] != 3:
        if image.shape[2] == 1:
            # Expand single channel to 3 channels
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] > 3:
            # Take first 3 channels
            image = image[:, :, :3]
    
    # Normalize to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() > 1.0:
            # Assume values are in 0-255 range or higher
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            # Assume values are normalized 0-1
            image = (image * 255).astype(np.uint8)
    
    # Final check: ensure exactly 3 dimensions
    assert image.ndim == 3, f"Image must have exactly 3 dimensions, got {image.ndim}"
    assert image.shape[2] == 3, f"Image must have exactly 3 channels, got {image.shape[2]}"
    
    return image


def save_detection_as_tif(
    detection_result: Any,
    original_array: np.ndarray,
    metadata: Dict[str, Any],
    output_path: Path,
    detector: Optional[Any] = None,
    temp_image_path: Optional[Path] = None,
    sam_image_shape: Optional[Tuple[int, int]] = None,
    original_image_shape: Optional[Tuple[int, int]] = None,
    is_sam_result: bool = False,
    verbose: bool = False
) -> None:
    """
    Save detection or SAM results as a georeferenced TIF file.
    
    Args:
        detection_result: YOLO detection result object, or list of SAM result objects if is_sam_result=True
        original_array: Original array data
        metadata: Metadata dictionary with transform, CRS, etc.
        output_path: Path to save the output TIF file
        detector: Detector instance for visualization (optional, only for detection results)
        temp_image_path: Path to temporary image file used for detection (optional)
        sam_image_shape: Shape of the image used for SAM inference (height, width) - only for SAM results
        original_image_shape: Shape of the original image (height, width) - only for SAM results
        is_sam_result: If True, detection_result is a list of SAM results
    """
    # Convert array to image format (this ensures it's in H, W, C format)
    image_array = array_to_image(original_array)
    
    # Get dimensions from the converted image array (which is in H, W, C format)
    # This is the correct way to get height and width regardless of original array format
    original_height, original_width = image_array.shape[:2]
    
    # Handle SAM results differently
    if is_sam_result:
        # Extract masks from SAM results
        sam_results = detection_result  # Rename for clarity
        all_masks = []
        for idx, sam_result in enumerate(sam_results):
            try:
                # SAM results from ultralytics might have masks in different formats
                # Try multiple ways to access masks
                masks = None
                
                # Method 1: Check if masks attribute exists
                if hasattr(sam_result, 'masks') and sam_result.masks is not None:
                    masks_data = sam_result.masks.data
                    if masks_data is not None:
                        # Convert to numpy if tensor
                        if hasattr(masks_data, 'cpu'):
                            masks = masks_data.cpu().numpy()
                        else:
                            masks = np.array(masks_data)
                
                # Method 2: Check if result has masks directly
                if masks is None and hasattr(sam_result, 'masks'):
                    try:
                        # Try accessing masks differently
                        if hasattr(sam_result.masks, 'xy'):
                            # SAM might store masks as polygons
                            pass
                        elif hasattr(sam_result.masks, 'data'):
                            masks_data = sam_result.masks.data
                            if masks_data is not None:
                                if hasattr(masks_data, 'cpu'):
                                    masks = masks_data.cpu().numpy()
                                else:
                                    masks = np.array(masks_data)
                    except:
                        pass
                
                # Method 3: Try to get masks from result directly
                if masks is None:
                    try:
                        # Check if result is a list/array of results
                        if isinstance(sam_result, (list, tuple)) and len(sam_result) > 0:
                            # Get first result if it's a list
                            if hasattr(sam_result[0], 'masks'):
                                masks_data = sam_result[0].masks.data
                                if masks_data is not None:
                                    if hasattr(masks_data, 'cpu'):
                                        masks = masks_data.cpu().numpy()
                                    else:
                                        masks = np.array(masks_data)
                    except:
                        pass
                
                if masks is not None:
                    # Handle different mask formats
                    if len(masks.shape) == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0, :, :]  # (N, 1, H, W) -> (N, H, W)
                    elif len(masks.shape) == 3:
                        # Already in (N, H, W) format
                        pass
                    elif len(masks.shape) == 2:
                        # Single mask: (H, W) -> (1, H, W)
                        masks = masks[np.newaxis, :, :]
                    else:
                        print(f"Warning: Unexpected mask shape {masks.shape} for SAM result {idx}")
                        continue
                    
                    # Normalize masks to [0, 1] if needed
                    for i in range(masks.shape[0]):
                        mask = masks[i].copy()
                        
                        # Check if mask is boolean and convert to float
                        if mask.dtype == bool:
                            mask = mask.astype(np.float32)
                        elif mask.dtype in [np.int8, np.int16, np.int32, np.int64]:
                            # Integer mask, normalize to [0, 1]
                            if mask.max() > 1.0:
                                mask = mask.astype(np.float32) / 255.0
                            else:
                                mask = mask.astype(np.float32)
                        else:
                            # Float mask
                            if mask.max() > 1.0:
                                mask = mask / 255.0
                        
                        mask = np.clip(mask, 0, 1)
                        
                        # Debug: print mask info
                        if verbose:
                            print(f"  SAM mask {idx}-{i}: shape={mask.shape}, dtype={mask.dtype}, min={mask.min():.3f}, max={mask.max():.3f}, non-zero={np.count_nonzero(mask)}")
                        
                        all_masks.append(mask)
                else:
                    print(f"Warning: No masks found in SAM result {idx}")
                    # Try to inspect the result structure
                    if verbose:
                        print(f"  SAM result {idx} attributes: {[a for a in dir(sam_result) if not a.startswith('_')]}")
                        if hasattr(sam_result, 'masks'):
                            print(f"  SAM result {idx} masks type: {type(sam_result.masks)}")
                            if hasattr(sam_result.masks, '__dict__'):
                                print(f"  SAM result {idx} masks attributes: {[a for a in dir(sam_result.masks) if not a.startswith('_')]}")
            except Exception as e:
                print(f"Warning: Failed to extract mask from SAM result {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Extracted {len(all_masks)} mask(s) from SAM results")
        
        if len(all_masks) == 0:
            print(f"Warning: No masks found in SAM results, creating empty TIF")
            all_masks = [np.zeros((original_height, original_width), dtype=np.float32)]
        
        # If SAM image was resized, resize masks back to original size
        if sam_image_shape is not None and original_image_shape is not None:
            sam_h, sam_w = sam_image_shape
            orig_h, orig_w = original_image_shape
            
            if sam_h != orig_h or sam_w != orig_w:
                print(f"Resizing masks from SAM image size {sam_w}x{sam_h} to original size {orig_w}x{orig_h}")
                resized_masks = []
                for mask in all_masks:
                    if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
                        mask_resized = cv2.resize(
                            mask.astype(np.float32),
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR
                        )
                        resized_masks.append(mask_resized)
                    else:
                        resized_masks.append(mask)
                all_masks = resized_masks
        
        # Create visualization with masks
        annotated_img = image_array.copy()
        mask_overlay = annotated_img.copy().astype(np.float32)
        
        # Generate distinct colors for each mask
        num_masks = len(all_masks)
        print(f"Visualizing {num_masks} mask(s) on image")
        
        for idx, mask in enumerate(all_masks):
            # Generate color using HSV
            hue = int(180 * idx / max(1, num_masks))
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = np.array([float(c) for c in color_bgr])
            
            # Ensure mask matches image dimensions
            if mask.shape[0] != original_height or mask.shape[1] != original_width:
                mask = cv2.resize(mask.astype(np.float32), (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            
            # Normalize mask to 0-1
            mask = np.clip(mask, 0, 1)
            
            # Debug: check mask values
            if verbose:
                print(f"  Mask {idx}: shape={mask.shape}, dtype={mask.dtype}, min={mask.min():.3f}, max={mask.max():.3f}, non-zero pixels={np.count_nonzero(mask)}")
            
            # Threshold mask to ensure binary-like behavior (values > 0.5 are considered mask)
            # This helps with float masks that might have low values
            mask_binary = (mask > 0.5).astype(np.float32)
            
            # Create colored mask overlay
            mask_3d = np.stack([mask_binary, mask_binary, mask_binary], axis=2)
            colored_mask = mask_3d * color
            
            # Blend mask with image
            alpha = 0.5  # Increased transparency for better visibility
            mask_overlay = mask_overlay * (1 - alpha * mask_3d) + colored_mask * (alpha * mask_3d)
        
        annotated_img = mask_overlay.astype(np.uint8)
        
    else:
        # Original detection result handling
        # Get visualized result
        # Try to use the detector's visualize method if we have a temp image path
        annotated_img = None
        if detector and temp_image_path and temp_image_path.exists():
            try:
                annotated_img = detector.visualize(temp_image_path, detection_result, None)
            except:
                pass
        
        # If visualize didn't work, try alternative approach
        if annotated_img is None or (hasattr(annotated_img, 'size') and annotated_img.size == 0):
            if hasattr(detection_result, 'plot'):
                try:
                    annotated_img = detection_result.plot()
                except:
                    annotated_img = None
            
            if annotated_img is None and hasattr(detection_result, 'render'):
                try:
                    annotated_img = detection_result.render()[0]
                    if len(annotated_img.shape) == 3:
                        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                except:
                    annotated_img = None
            
            # Fallback: try to manually draw boxes and masks on image for merged results
            if annotated_img is None and hasattr(detection_result, 'boxes') and detection_result.boxes is not None:
                try:
                    # Get boxes and draw them manually
                    boxes = detection_result.boxes.xyxy.cpu().numpy() if hasattr(detection_result.boxes.xyxy, 'cpu') else detection_result.boxes.xyxy
                    confidences = detection_result.boxes.conf.cpu().numpy() if hasattr(detection_result.boxes.conf, 'cpu') else detection_result.boxes.conf
                    
                    # Create a copy of the image to draw on
                    annotated_img = image_array.copy()
                    
                    # Get masks if available (for instance segmentation)
                    masks = None
                    if hasattr(detection_result, 'masks') and detection_result.masks is not None:
                        try:
                            masks_data = detection_result.masks.data
                            if masks_data is not None:
                                # Convert to numpy if tensor
                                if hasattr(masks_data, 'cpu'):
                                    masks = masks_data.cpu().numpy()
                                else:
                                    masks = np.array(masks_data)
                                # Handle different mask formats
                                if len(masks.shape) == 4 and masks.shape[1] == 1:
                                    masks = masks[:, 0, :, :]  # Remove channel dimension
                                elif len(masks.shape) == 3:
                                    # Already in (N, H, W) format
                                    pass
                                else:
                                    masks = None
                        except:
                            masks = None
                    
                    # Draw masks first (so boxes appear on top)
                    if masks is not None and len(masks) > 0:
                        # Create a colored overlay for masks
                        mask_overlay = annotated_img.copy().astype(np.float32)
                        
                        for i, mask in enumerate(masks):
                            if i >= len(boxes):
                                break
                            
                            # Resize mask to image size if needed
                            if mask.shape[0] != original_height or mask.shape[1] != original_width:
                                mask = cv2.resize(mask.astype(np.float32), (original_width, original_height), interpolation=cv2.INTER_LINEAR)
                            
                            # Normalize mask to 0-1 range
                            if mask.max() > 1.0:
                                mask = mask / 255.0
                            mask = np.clip(mask, 0, 1)
                            
                            # Create colored mask (green with transparency)
                            color = np.array([0, 255, 0])  # Green color
                            mask_3d = np.stack([mask, mask, mask], axis=2)
                            colored_mask = mask_3d * color
                            
                            # Blend mask with image
                            alpha = 0.3  # Transparency
                            mask_overlay = mask_overlay * (1 - alpha * mask_3d) + colored_mask * (alpha * mask_3d)
                        
                        annotated_img = mask_overlay.astype(np.uint8)
                    
                    # Draw bounding boxes
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = map(int, box[:4])
                        # Draw rectangle
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw confidence score
                        label = f'{conf:.2f}'
                        cv2.putText(annotated_img, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    # If manual drawing fails, use original image
                    print(f"Warning: Failed to draw detections manually: {e}")
                    annotated_img = image_array
        
        # Final fallback: use original image if all else fails
        if annotated_img is None:
            annotated_img = image_array
    
    # Ensure annotated_img is not None and has valid shape
    if annotated_img is None:
        annotated_img = image_array
    
    # Ensure annotated image matches original dimensions
    if annotated_img is not None and annotated_img.shape[:2] != (original_height, original_width):
        annotated_img = cv2.resize(
            annotated_img,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Convert BGR to RGB for rasterio
    if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    else:
        annotated_img_rgb = annotated_img
    
    # Prepare for rasterio: (channels, height, width)
    if len(annotated_img_rgb.shape) == 2:
        annotated_img_rgb = np.expand_dims(annotated_img_rgb, axis=2)
    if len(annotated_img_rgb.shape) == 3:
        annotated_img_rgb = np.transpose(annotated_img_rgb, (2, 0, 1))
    
    # Get transform from metadata or create from bounds
    transform = metadata.get('transform')
    if transform is None and 'bounds' in metadata:
        bounds = metadata['bounds']
        transform = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            original_width, original_height
        )
    elif transform is None:
        # Create identity transform as fallback
        transform = Affine.identity()
    
    # Get CRS from metadata or default to EPSG:4326
    crs = metadata.get('crs', CRS.from_epsg(4326))
    
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
    
    if 'nodata' in metadata:
        output_profile['nodata'] = metadata['nodata']
    
    # Write GeoTIFF
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(annotated_img_rgb)
        dst.crs = crs
        dst.transform = transform
        
        # Add metadata tags
        tags = {k: v for k, v in metadata.items() 
               if k not in ['transform', 'crs', 'nodata', 'bounds']}
        if is_sam_result:
            # Count masks from SAM results
            mask_count = 0
            if isinstance(detection_result, list):
                for sam_res in detection_result:
                    if hasattr(sam_res, 'masks') and sam_res.masks is not None:
                        if hasattr(sam_res.masks, 'data') and sam_res.masks.data is not None:
                            mask_count += len(sam_res.masks.data) if hasattr(len, '__call__') else 1
            tags['sam_masks_count'] = mask_count
        if tags:
            dst.update_tags(**tags)
    
    result_type = "SAM results" if is_sam_result else "detection result"
    print(f"Saved {result_type} to: {output_path}")

