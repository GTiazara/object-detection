"""SAM (Segment Anything Model) utilities.

This module provides utility functions for preparing instance segmentation results
for use with SAM, including extracting masks and generating points inside masks.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import cv2


def extract_masks_from_result(detection_result: Any) -> List[np.ndarray]:
    """
    Extract masks from instance segmentation result.
    
    Args:
        detection_result: Detection result object with masks attribute
            (e.g., YOLO result with result.masks.data)
        
    Returns:
        List of mask arrays, each of shape (H, W) with values in [0, 1]
    """
    masks = []
    
    if not hasattr(detection_result, 'masks') or detection_result.masks is None:
        return masks
    
    try:
        masks_data = detection_result.masks.data
        if masks_data is None:
            return masks
        
        # Convert to numpy if tensor
        if hasattr(masks_data, 'cpu'):
            masks_array = masks_data.cpu().numpy()
        else:
            masks_array = np.array(masks_data)
        
        # Handle different mask formats
        if len(masks_array.shape) == 4 and masks_array.shape[1] == 1:
            # Shape: (N, 1, H, W) -> (N, H, W)
            masks_array = masks_array[:, 0, :, :]
        elif len(masks_array.shape) == 3:
            # Shape: (N, H, W) - already correct
            pass
        elif len(masks_array.shape) == 2:
            # Single mask: (H, W) -> (1, H, W)
            masks_array = masks_array[np.newaxis, :, :]
        else:
            return masks
        
        # Normalize masks to [0, 1] range if needed
        for i in range(masks_array.shape[0]):
            mask = masks_array[i]
            # Normalize if values are not in [0, 1]
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = np.clip(mask, 0, 1)
            masks.append(mask)
            
    except Exception as e:
        print(f"Warning: Failed to extract masks: {e}")
        return masks
    
    return masks


def generate_points_in_mask(
    mask: np.ndarray,
    num_points: Optional[int] = None,
    min_points: int = 1,
    max_points: int = 64,
    strategy: str = "uniform"
) -> List[Tuple[int, int]]:
    """
    Generate points inside a mask for SAM segmentation.
    
    Args:
        mask: Binary mask array of shape (H, W) with values in [0, 1]
        num_points: Number of points to generate. If None, automatically determined
            based on mask area
        min_points: Minimum number of points to generate
        max_points: Maximum number of points to generate
        strategy: Strategy for point generation
            - "uniform": Uniformly distributed points
            - "random": Random points within mask
            - "grid": Grid-based sampling within mask
            - "adaptive": Adaptive based on mask area (default behavior if num_points is None)
    
    Returns:
        List of (x, y) coordinate tuples (points inside the mask)
    """
    # Ensure mask is binary
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Get mask area
    mask_area = np.sum(mask_binary)
    
    if mask_area == 0:
        return []
    
    # Determine number of points if not specified
    if num_points is None:
        # Adaptive: scale points based on mask area
        # Use square root of area as base, scaled appropriately
        area_ratio = mask_area / (mask_binary.shape[0] * mask_binary.shape[1])
        base_points = int(np.sqrt(mask_area) * 0.1)  # Scale factor can be adjusted
        num_points = max(min_points, min(max_points, base_points))
    else:
        num_points = max(min_points, min(max_points, num_points))
    
    # Get coordinates of all mask pixels
    y_coords, x_coords = np.where(mask_binary > 0)
    
    if len(x_coords) == 0:
        return []
    
    points = []
    
    if strategy == "random":
        # Random sampling
        if len(x_coords) <= num_points:
            # If we have fewer pixels than requested points, return all
            points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
        else:
            # Randomly sample points
            indices = np.random.choice(len(x_coords), size=num_points, replace=False)
            points = [(int(x_coords[i]), int(y_coords[i])) for i in indices]
    
    elif strategy == "grid":
        # Grid-based sampling
        # Create a grid and sample points that fall within the mask
        h, w = mask_binary.shape
        grid_size = int(np.ceil(np.sqrt(num_points)))
        step_y = max(1, h // grid_size)
        step_x = max(1, w // grid_size)
        
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                if mask_binary[y, x] > 0:
                    points.append((int(x), int(y)))
                    if len(points) >= num_points:
                        break
            if len(points) >= num_points:
                break
    
    elif strategy == "uniform" or strategy == "adaptive":
        # Uniform distribution: try to spread points evenly across the mask
        if len(x_coords) <= num_points:
            # If we have fewer pixels than requested points, return all
            points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
        else:
            # Use k-means-like approach or spatial sampling
            # Simple approach: divide mask into regions and sample from each
            if num_points == 1:
                # Return centroid
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))
                points = [(cx, cy)]
            else:
                # Divide into grid regions and sample from each
                h, w = mask_binary.shape
                grid_size = int(np.ceil(np.sqrt(num_points)))
                cell_h = h / grid_size
                cell_w = w / grid_size
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        y_min = int(i * cell_h)
                        y_max = int((i + 1) * cell_h)
                        x_min = int(j * cell_w)
                        x_max = int((j + 1) * cell_w)
                        
                        # Find pixels in this cell that are in the mask
                        cell_mask = (y_coords >= y_min) & (y_coords < y_max) & \
                                   (x_coords >= x_min) & (x_coords < x_max)
                        cell_x = x_coords[cell_mask]
                        cell_y = y_coords[cell_mask]
                        
                        if len(cell_x) > 0:
                            # Sample one point from this cell (preferably near center)
                            if len(cell_x) == 1:
                                points.append((int(cell_x[0]), int(cell_y[0])))
                            else:
                                # Use centroid of pixels in this cell
                                cx = int(np.mean(cell_x))
                                cy = int(np.mean(cell_y))
                                points.append((cx, cy))
                        
                        if len(points) >= num_points:
                            break
                    if len(points) >= num_points:
                        break
                
                # If we still need more points, randomly sample from remaining
                if len(points) < num_points:
                    remaining = num_points - len(points)
                    # Get all points not yet selected
                    selected_coords = set(points)
                    available = [(x, y) for x, y in zip(x_coords, y_coords) 
                                if (int(x), int(y)) not in selected_coords]
                    if len(available) > 0:
                        indices = np.random.choice(len(available), 
                                                 size=min(remaining, len(available)), 
                                                 replace=False)
                        points.extend([available[i] for i in indices])
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of: 'uniform', 'random', 'grid', 'adaptive'")
    
    return points


def generate_points_outside_mask(
    mask: np.ndarray,
    num_points: Optional[int] = None,
    min_points: int = 1,
    max_points: int = 64,
    strategy: str = "uniform"
) -> List[Tuple[int, int]]:
    """
    Generate points outside a mask for SAM segmentation (negative points).
    
    Args:
        mask: Binary mask array of shape (H, W) with values in [0, 1]
        num_points: Number of points to generate. If None, automatically determined
            based on background area
        min_points: Minimum number of points to generate
        max_points: Maximum number of points to generate
        strategy: Strategy for point generation ('uniform', 'random', 'grid', 'adaptive')
    
    Returns:
        List of (x, y) coordinate tuples (points outside the mask)
    """
    # Ensure mask is binary
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Get background area (outside mask)
    background_binary = 1 - mask_binary
    background_area = np.sum(background_binary)
    
    if background_area == 0:
        return []
    
    # Determine number of points if not specified
    if num_points is None:
        # Adaptive: scale points based on background area
        area_ratio = background_area / (background_binary.shape[0] * background_binary.shape[1])
        base_points = int(np.sqrt(background_area) * 0.1)  # Scale factor can be adjusted
        num_points = max(min_points, min(max_points, base_points))
    else:
        num_points = max(min_points, min(max_points, num_points))
    
    # Get coordinates of all background pixels
    y_coords, x_coords = np.where(background_binary > 0)
    
    if len(x_coords) == 0:
        return []
    
    points = []
    
    if strategy == "random":
        # Random sampling
        if len(x_coords) <= num_points:
            # If we have fewer pixels than requested points, return all
            points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
        else:
            # Randomly sample points
            indices = np.random.choice(len(x_coords), size=num_points, replace=False)
            points = [(int(x_coords[i]), int(y_coords[i])) for i in indices]
    
    elif strategy == "grid":
        # Grid-based sampling
        h, w = background_binary.shape
        grid_size = int(np.ceil(np.sqrt(num_points)))
        step_y = max(1, h // grid_size)
        step_x = max(1, w // grid_size)
        
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                if background_binary[y, x] > 0:
                    points.append((int(x), int(y)))
                    if len(points) >= num_points:
                        break
            if len(points) >= num_points:
                break
    
    elif strategy == "uniform" or strategy == "adaptive":
        # Uniform distribution: try to spread points evenly across the background
        if len(x_coords) <= num_points:
            # If we have fewer pixels than requested points, return all
            points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
        else:
            # Divide into grid regions and sample from each
            h, w = background_binary.shape
            grid_size = int(np.ceil(np.sqrt(num_points)))
            cell_h = h / grid_size
            cell_w = w / grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y_min = int(i * cell_h)
                    y_max = int((i + 1) * cell_h)
                    x_min = int(j * cell_w)
                    x_max = int((j + 1) * cell_w)
                    
                    # Find pixels in this cell that are in the background
                    cell_mask = (y_coords >= y_min) & (y_coords < y_max) & \
                               (x_coords >= x_min) & (x_coords < x_max)
                    cell_x = x_coords[cell_mask]
                    cell_y = y_coords[cell_mask]
                    
                    if len(cell_x) > 0:
                        # Sample one point from this cell
                        if len(cell_x) == 1:
                            points.append((int(cell_x[0]), int(cell_y[0])))
                        else:
                            # Use centroid of pixels in this cell
                            cx = int(np.mean(cell_x))
                            cy = int(np.mean(cell_y))
                            points.append((cx, cy))
                    
                    if len(points) >= num_points:
                        break
                if len(points) >= num_points:
                    break
            
            # If we still need more points, randomly sample from remaining
            if len(points) < num_points:
                remaining = num_points - len(points)
                selected_coords = set(points)
                available = [(x, y) for x, y in zip(x_coords, y_coords) 
                            if (int(x), int(y)) not in selected_coords]
                if len(available) > 0:
                    indices = np.random.choice(len(available), 
                                             size=min(remaining, len(available)), 
                                             replace=False)
                    points.extend([available[i] for i in indices])
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of: 'uniform', 'random', 'grid', 'adaptive'")
    
    return points


def extract_masks_and_points(
    detection_result: Any,
    num_points_per_mask: Optional[int] = None,
    min_points: int = 1,
    max_points: int = 64,
    strategy: str = "uniform"
) -> List[Dict[str, Any]]:
    """
    Extract masks from instance segmentation result and generate points for each mask.
    
    This function generates both positive points (inside masks, label=1) and negative points
    (outside masks, label=0) for SAM segmentation. This is necessary because SAM uses labels
    to distinguish between points that should be included (1) and excluded (0) from segmentation.
    
    Args:
        detection_result: Detection result object with masks and boxes attributes
        num_points_per_mask: Number of points to generate per mask. If None, automatically
            determined based on mask area
        min_points: Minimum number of points per mask
        max_points: Maximum number of points per mask
        strategy: Strategy for point generation ('uniform', 'random', 'grid', 'adaptive')
    
    Returns:
        List of dictionaries, each containing:
            - 'mask': numpy array of shape (H, W)
            - 'points': List of (x, y) tuples (positive points inside this specific mask)
            - 'mask_index': Index of the mask in the original result
            - 'class_label': Numeric class label from YOLO model (if available)
        Additionally, a special entry with key 'fused' contains:
            - 'fused_points': List of all (x, y) tuples (both positive and negative points)
            - 'fused_labels': List of labels (1 for positive points inside masks, 0 for negative points outside masks)
    """
    masks = extract_masks_from_result(detection_result)
    
    # Extract class labels from YOLO result
    class_labels = []
    if hasattr(detection_result, 'boxes') and detection_result.boxes is not None:
        try:
            if hasattr(detection_result.boxes, 'cls'):
                cls_data = detection_result.boxes.cls
                # Convert to numpy if tensor
                if hasattr(cls_data, 'cpu'):
                    class_labels = cls_data.cpu().numpy().astype(int).tolist()
                else:
                    class_labels = np.array(cls_data).astype(int).tolist()
        except Exception as e:
            print(f"Warning: Failed to extract class labels: {e}")
            class_labels = []
    
    # If we don't have class labels, use mask indices as fallback
    if len(class_labels) == 0:
        class_labels = list(range(len(masks)))
    
    results = []
    fused_points = []
    fused_labels = []
    
    # Combine all masks to create a combined mask for negative point generation
    combined_mask = None
    if len(masks) > 0:
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            # Normalize mask to binary
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask_binary = (mask > 0.5).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask_binary)
    
    for idx, mask in enumerate(masks):
        # Generate positive points inside mask (label=1 for SAM)
        positive_points = generate_points_in_mask(
            mask,
            num_points=num_points_per_mask,
            min_points=min_points,
            max_points=max_points,
            strategy=strategy
        )
        
        # Get class label for this mask (use mask index if class label not available)
        class_label = class_labels[idx] if idx < len(class_labels) else idx
        
        # Add positive points to fused list with label=1 (for SAM: positive point)
        for point in positive_points:
            fused_points.append(point)
            fused_labels.append(1)  # Label 1 = positive point (inside mask) for SAM
        
        results.append({
            'mask': mask,
            'points': positive_points,
            'mask_index': idx,
            'class_label': int(class_label)
        })
    
    # Generate negative points outside all masks (label=0 for SAM)
    if combined_mask is not None and len(masks) > 0:
        # Generate negative points outside the combined mask
        negative_points = generate_points_outside_mask(
            combined_mask,
            num_points=num_points_per_mask,  # Same number as positive points
            min_points=min_points,
            max_points=max_points,
            strategy=strategy
        )
        
        # Add negative points to fused list with label=0 (for SAM: negative point)
        for point in negative_points:
            fused_points.append(point)
            fused_labels.append(0)  # Label 0 = negative point (outside mask) for SAM
    
    # Add fused points and labels as a special entry
    if len(fused_points) > 0:
        results.append({
            'fused_points': fused_points,
            'fused_labels': fused_labels,
            'mask_index': -1  # Special marker for fused entry
        })
    
    return results


def format_points_for_sam(
    points: List[Tuple[int, int]],
    labels: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Format points for SAM input.
    
    SAM expects points as numpy arrays with shape (N, 2) where N is the number of points,
    and labels as numpy array with shape (N,) where 1 indicates positive point and 0 indicates negative.
    
    Args:
        points: List of (x, y) coordinate tuples
        labels: Optional list of labels (1 for positive, 0 for negative).
            If None, all points are treated as positive (label=1)
    
    Returns:
        Dictionary with:
            - 'points': numpy array of shape (N, 2) with coordinates
            - 'labels': numpy array of shape (N,) with labels
    """
    if len(points) == 0:
        return {
            'points': np.array([], dtype=np.float32).reshape(0, 2),
            'labels': np.array([], dtype=np.int32)
        }
    
    points_array = np.array(points, dtype=np.float32)
    
    if labels is None:
        labels_array = np.ones(len(points), dtype=np.int32)
    else:
        if len(labels) != len(points):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of points ({len(points)})")
        labels_array = np.array(labels, dtype=np.int32)
    
    return {
        'points': points_array,
        'labels': labels_array
    }


def save_image_with_points(
    image: np.ndarray,
    sam_data: List[Dict[str, Any]],
    output_path: Path,
    point_radius: int = 3,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    mask_alpha: float = 0.3,
    mask_colors: Optional[List[Tuple[int, int, int]]] = None,
    show_mask: bool = True,
    input_format: str = "RGB"
) -> None:
    """
    Save an image with SAM points and masks visualized.
    
    Args:
        image: Input image array of shape (H, W, C)
        sam_data: List of dictionaries from extract_masks_and_points, each containing:
            - 'mask': numpy array of shape (H, W)
            - 'points': List of (x, y) tuples
            - 'mask_index': Index of the mask
        output_path: Path where to save the visualization
        point_radius: Radius of the circles drawn for points
        point_color: BGR color tuple for points (default: green)
        mask_alpha: Transparency of mask overlay (0.0 to 1.0)
        mask_colors: Optional list of BGR colors for each mask. If None, uses default colors
        show_mask: Whether to show mask overlay (default: True)
        input_format: Color format of input image - "RGB" or "BGR" (default: "RGB")
    """
    # Make a copy of the image to draw on
    vis_image = image.copy()
    
    # Ensure image is in uint8 format with values in [0, 255]
    if vis_image.dtype != np.uint8:
        if vis_image.max() <= 1.0:
            # Image is normalized [0, 1], scale to [0, 255]
            vis_image = (vis_image * 255).astype(np.uint8)
        else:
            # Image is in [0, 255] or higher, clip and convert
            vis_image = np.clip(vis_image, 0, 255).astype(np.uint8)
    
    # Convert to BGR format for OpenCV (cv2.imwrite expects BGR)
    if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
        if input_format.upper() == "RGB":
            # Convert RGB to BGR for OpenCV
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        # If already BGR, no conversion needed
    
    # Generate default colors if not provided
    if mask_colors is None:
        # Generate distinct colors for each mask
        num_masks = len(sam_data)
        mask_colors = []
        for i in range(num_masks):
            # Generate colors using HSV and convert to BGR
            hue = int(180 * i / max(1, num_masks))
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            mask_colors.append(tuple(int(c) for c in color_bgr))
    
    # Draw masks first (if enabled)
    if show_mask:
        for idx, sam_item in enumerate(sam_data):
            mask = sam_item['mask']
            color = mask_colors[idx % len(mask_colors)]
            
            # Ensure mask is binary and same size as image
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Resize mask if needed
            if mask_binary.shape[:2] != vis_image.shape[:2]:
                mask_binary = cv2.resize(mask_binary, 
                                       (vis_image.shape[1], vis_image.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            
            # Create colored mask
            mask_colored = np.zeros_like(vis_image)
            mask_colored[mask_binary > 0] = color
            
            # Blend mask with image
            vis_image = cv2.addWeighted(vis_image, 1.0 - mask_alpha, 
                                       mask_colored, mask_alpha, 0)
    
    # Draw points on top
    for idx, sam_item in enumerate(sam_data):
        points = sam_item['points']
        color = mask_colors[idx % len(mask_colors)] if show_mask else point_color
        
        for point in points:
            x, y = int(point[0]), int(point[1])
            # Draw filled circle
            cv2.circle(vis_image, (x, y), point_radius, color, -1)
            # Draw outline for better visibility
            cv2.circle(vis_image, (x, y), point_radius, (0, 0, 0), 1)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the image
    cv2.imwrite(str(output_path), vis_image)
    print(f"Saved SAM visualization to: {output_path}")

