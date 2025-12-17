"""Simple SAM (Segment Anything Model) prediction module.

This module provides a simple interface to run SAM inference on images
and save the results as TIF files.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

try:
    from ultralytics import SAM
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    SAM = None


def predict_sam(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = "sam2.1_b.pt",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Path:
    """
    Run SAM inference on an image and save the result as a TIF file.
    
    Args:
        image_path: Path to the input image file
        model_path: Path to the SAM model file (default: "sam2.1_b.pt")
        output_path: Path to save the output TIF file. If None, saves next to input with "_sam_results.tif" suffix
        verbose: Print detailed information during processing
        
    Returns:
        Path to the saved output TIF file
        
    Raises:
        ImportError: If ultralytics SAM is not available
        FileNotFoundError: If image or model file doesn't exist
    """
    if not SAM_AVAILABLE:
        raise ImportError("ultralytics SAM is not available. Please install ultralytics: pip install ultralytics")
    
    # Convert to Path objects
    image_path = Path(image_path)
    model_path = Path(model_path)
    
    # Check if files exist
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"SAM model file not found: {model_path}")
    
    # Determine output path
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_sam_results.tif"
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Loading SAM model from: {model_path}")
    
    # Load SAM model
    model = SAM(str(model_path))
    
    if verbose:
        model.info()
        print(f"Processing image: {image_path}")
    
    # Load image
    # Try to load with rasterio first (for georeferenced TIF files)
    metadata = {}
    try:
        with rasterio.open(image_path) as src:
            image_array = src.read()
            # Get metadata for later use
            metadata['transform'] = src.transform
            metadata['crs'] = src.crs
            metadata['nodata'] = src.nodata
            
            # Convert from (C, H, W) to (H, W, C) for processing
            if len(image_array.shape) == 3:
                image_array = np.transpose(image_array, (1, 2, 0))
            
            # Handle different channel counts
            if image_array.shape[2] == 1:
                # Grayscale: convert to 3-channel
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[2] == 4:
                # RGBA: use RGB channels
                image_array = image_array[:, :, :3]
            elif image_array.shape[2] > 3:
                # More than 3 channels: use first 3
                image_array = image_array[:, :, :3]
            
            # Normalize to uint8 if needed
            if image_array.dtype != np.uint8:
                if image_array.max() > 1.0:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                else:
                    image_array = (image_array * 255).astype(np.uint8)
    except Exception as e:
        # Fallback to OpenCV for regular images
        if verbose:
            print(f"Could not load with rasterio ({e}), trying OpenCV...")
        image_array = cv2.imread(str(image_path))
        if image_array is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for consistency
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    original_shape = image_array.shape[:2]  # (height, width)
    
    if verbose:
        print(f"Image shape: {image_array.shape}")
    
    # Run SAM inference
    # Convert image to format expected by SAM (PIL Image or numpy array)
    # SAM expects RGB format
    results = model(image_array)
    
    if verbose:
        print(f"SAM inference completed. Number of results: {len(results)}")
    
    # Save results as TIF
    # Use the save_detection_as_tif utility if available, otherwise save manually
    try:
        from src.utils.image_utils import save_detection_as_tif
        
        # Prepare metadata if not already set
        if 'transform' not in metadata:
            metadata['transform'] = Affine.identity()
        if 'crs' not in metadata:
            metadata['crs'] = CRS.from_epsg(4326)
        
        # Save using the utility function
        save_detection_as_tif(
            detection_result=results,
            original_array=image_array,
            metadata=metadata,
            output_path=output_path,
            is_sam_result=True,
            verbose=verbose
        )
    except ImportError:
        # Fallback: save manually if utility is not available
        if verbose:
            print("Warning: Could not import save_detection_as_tif, saving manually...")
        
        # Extract masks from results
        all_masks = []
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks_data = result.masks.data
                if masks_data is not None:
                    if hasattr(masks_data, 'cpu'):
                        masks = masks_data.cpu().numpy()
                    else:
                        masks = np.array(masks_data)
                    
                    if len(masks.shape) == 3:
                        all_masks.extend([masks[i] for i in range(masks.shape[0])])
                    elif len(masks.shape) == 2:
                        all_masks.append(masks)
        
        # Create visualization
        if len(all_masks) > 0:
            annotated_img = image_array.copy().astype(np.float32)
            for idx, mask in enumerate(all_masks):
                # Resize mask if needed
                if mask.shape != original_shape:
                    mask = cv2.resize(mask.astype(np.float32), 
                                    (original_shape[1], original_shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
                
                # Normalize mask
                if mask.max() > 1.0:
                    mask = mask / 255.0
                mask = np.clip(mask, 0, 1)
                
                # Create colored overlay
                color = np.array([0, 255, 0])  # Green
                mask_3d = np.stack([mask, mask, mask], axis=2)
                colored_mask = mask_3d * color
                
                # Blend
                alpha = 0.5
                annotated_img = annotated_img * (1 - alpha * mask_3d) + colored_mask * (alpha * mask_3d)
            
            annotated_img = annotated_img.astype(np.uint8)
        else:
            annotated_img = image_array
        
        # Convert RGB to BGR for OpenCV (if needed) or keep RGB for rasterio
        # Rasterio expects RGB, so we keep it as is
        
        # Prepare for rasterio: (channels, height, width)
        if len(annotated_img.shape) == 3:
            annotated_img_rgb = np.transpose(annotated_img, (2, 0, 1))
        else:
            annotated_img_rgb = np.expand_dims(annotated_img, axis=0)
        
        # Get transform and CRS from metadata or use defaults
        transform = metadata.get('transform', Affine.identity())
        crs = metadata.get('crs', CRS.from_epsg(4326))
        
        # Write GeoTIFF
        output_profile = {
            'driver': 'GTiff',
            'height': original_shape[0],
            'width': original_shape[1],
            'count': annotated_img_rgb.shape[0],
            'dtype': annotated_img_rgb.dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
            'tiled': True,
        }
        
        if 'nodata' in metadata:
            output_profile['nodata'] = metadata['nodata']
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(annotated_img_rgb)
            dst.crs = crs
            dst.transform = transform
        
        if verbose:
            print(f"Saved SAM results to: {output_path}")
    
    return output_path


def main():
    """Command-line interface for SAM prediction."""
    
    try:
        output_path = predict_sam(
            image_path="/home/GTiazara/Documents/workspace/get_experience_project/geo-dataset-builder/output/0_1.tif",
            model_path="/home/GTiazara/Documents/workspace/get_experience_project/object-detection/sam_b.pt",
            output_path="/home/GTiazara/Documents/workspace/get_experience_project/object-detection/data/output/0_1_sam_results.tif",
            verbose=True
        )
        print(f"\nSuccess! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())#exit the program if it fails
    main()

