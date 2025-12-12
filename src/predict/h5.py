"""Module for predicting objects in H5/HDF5 files.

This module reads H5 files, walks through all groups to find array datasets,
extracts metadata from dataset attributes (transform, CRS, etc.), runs prediction,
and saves results as georeferenced TIF files.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import defaultdict
import numpy as np
import h5py
import cv2
import rasterio
import re
import sys
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS

import albumentations as A

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent

import yaml

from src.model_loader import ModelLoader
from src.detector import Detector
from src.utils.tile import split_image_into_tiles, merge_tile_detections, TileProcessor
from src.utils.image_utils import array_to_image, save_detection_as_tif
from src.utils.sam_utils import extract_masks_and_points, format_points_for_sam, save_image_with_points

try:
    from ultralytics import SAM
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    SAM = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses conf_h5.yaml in project root.
        
    Returns:
        Dictionary containing configuration values with defaults.
    """
    if config_path is None:
        config_path = project_root / "conf_h5.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration")
        return _get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # Merge with defaults to ensure all keys exist
        default_config = _get_default_config()
        return _merge_config(default_config, config)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        print("Using default configuration")
        return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        'model': {
            'local_path': None,
            'model_variant': 'training',
            'force_download': False
        },
        'detection': {
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'image_size': 640,
            'resize': {
                'enabled': False,
                'width': None,
                'height': None,
                'interpolation': 'bilinear'
            },
            'tiling': {
                'enabled': False,
                'tile_size': None,
                'overlap': 0,
                'merge_nms_iou': 0.45
            }
        },
        'paths': {
            'input_path': None,
            'output_dir': None
        },
        'processing': {
            'continue_on_error': True,
            'verbose': True
        },
        'h5': {
            'min_dimensions': 2,
            'max_dimensions': 4,
            'include_patterns': [],
            'exclude_patterns': [],
            'metadata': {
                'transform_attr': 'transform',
                'crs_attr': 'crs',
                'nodata_attr': ['nodata', 'no_data', 'nodata_value'],
                'bounds_attr': 'bounds'
            }
        },
        'output': {
            'save': True,
            'filename_pattern': '{h5_stem}_{dataset_path}_detections.tif'
        },
        'sam': {
            'enabled': False,
            'model_path': None,  # e.g., 'sam2.1_b.pt'
            'use_original_image': True,  # Use original array instead of processed image_array
            'resize': {
                'enabled': False,
                'width': None,  # Target width (if None, maintains aspect ratio based on height)
                'height': None,  # Target height (if None, maintains aspect ratio based on width)
                'interpolation': 'bilinear'  # Options: 'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'
            }
        }
    }


def _merge_config(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge user config into default config."""
    result = default.copy()
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def should_process_dataset(dataset_path: str, config: Dict[str, Any]) -> bool:
    """
    Check if a dataset should be processed based on include/exclude patterns.
    
    Args:
        dataset_path: Path to the dataset in H5 file
        config: Configuration dictionary
        
    Returns:
        True if dataset should be processed, False otherwise
    """
    h5_config = config.get('h5', {})
    include_patterns = h5_config.get('include_patterns', [])
    exclude_patterns = h5_config.get('exclude_patterns', [])
    
    # If include patterns are specified, dataset must match at least one
    if include_patterns:
        matches_include = any(
            re.search(pattern, dataset_path) for pattern in include_patterns
        )
        if not matches_include:
            return False
    
    # Dataset must not match any exclude pattern
    if exclude_patterns:
        matches_exclude = any(
            re.search(pattern, dataset_path) for pattern in exclude_patterns
        )
        if matches_exclude:
            return False
    
    return True


def walk_groups(h5_file: h5py.File, base_path: str = "/", config: Optional[Dict[str, Any]] = None) -> List[Tuple[str, h5py.Dataset]]:
    """
    Walk through all groups in an H5 file and find array datasets.
    
    Args:
        h5_file: Open H5 file object
        base_path: Base path for recursion (default: "/")
        config: Configuration dictionary (optional, for filtering)
        
    Returns:
        List of tuples (path, dataset) for all array datasets found
    """
    datasets = []
    h5_config = config.get('h5', {}) if config else {}
    min_dim = h5_config.get('min_dimensions', 2)
    max_dim = h5_config.get('max_dimensions', 4)
    
    def _walk(name, obj):
        """Recursive function to walk through groups."""
        if isinstance(obj, h5py.Dataset):
            # Check if it's an array dataset (not a scalar) and meets dimension requirements
            if obj.ndim >= min_dim and obj.ndim <= max_dim:
                # Check include/exclude patterns if config is provided
                if config is None or should_process_dataset(name, config):
                    datasets.append((name, obj))
        elif isinstance(obj, h5py.Group):
            # Recursively walk through subgroups
            for key in obj.keys():
                _walk(f"{name}/{key}" if name != "/" else f"/{key}", obj[key])
    
    # Start walking from root
    for key in h5_file.keys():
        _walk(f"/{key}" if base_path == "/" else f"{base_path}/{key}", h5_file[key])
    
    return datasets


def extract_metadata(dataset: h5py.Dataset, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract metadata from dataset attributes.
    
    Args:
        dataset: H5 dataset object
        config: Configuration dictionary (optional, for attribute name customization)
        
    Returns:
        Dictionary containing metadata (transform, CRS, nodata, etc.)
    """
    metadata = {}
    attrs = dataset.attrs
    
    # Get attribute names from config or use defaults
    h5_config = config.get('h5', {}) if config else {}
    metadata_config = h5_config.get('metadata', {})
    
    # Helper function to get attribute value by name(s)
    def get_attr_value(attr_names):
        """Get attribute value, supporting both single name and list of names."""
        if isinstance(attr_names, str):
            attr_names = [attr_names]
        for name in attr_names:
            if name in attrs:
                return attrs[name]
        return None
    
    # Extract transform information
    transform_attr = metadata_config.get('transform_attr', 'transform')
    transform_data = get_attr_value(transform_attr)
    if transform_data is not None:
        if isinstance(transform_data, (list, tuple, np.ndarray)) and len(transform_data) >= 6:
            # Affine transform parameters: [a, b, c, d, e, f]
            metadata['transform'] = Affine(*transform_data[:6])
        elif isinstance(transform_data, bytes):
            # Try to decode if it's stored as bytes
            try:
                transform_data = np.frombuffer(transform_data, dtype=np.float64)
                if len(transform_data) >= 6:
                    metadata['transform'] = Affine(*transform_data[:6])
            except:
                pass
    
    # Extract CRS information
    crs_attr = metadata_config.get('crs_attr', 'crs')
    crs_data = get_attr_value(crs_attr)
    if crs_data is not None:
        if isinstance(crs_data, (str, bytes)):
            try:
                if isinstance(crs_data, bytes):
                    crs_data = crs_data.decode('utf-8')
                metadata['crs'] = CRS.from_string(crs_data)
            except:
                pass
        elif isinstance(crs_data, (int, np.integer)):
            # EPSG code
            try:
                metadata['crs'] = CRS.from_epsg(int(crs_data))
            except:
                pass
    
    # Extract nodata value
    nodata_attr = metadata_config.get('nodata_attr', ['nodata', 'no_data', 'nodata_value'])
    nodata_value = get_attr_value(nodata_attr)
    if nodata_value is not None:
        try:
            metadata['nodata'] = float(nodata_value)
        except:
            pass
    
    # Extract bounds if available
    bounds_attr = metadata_config.get('bounds_attr', 'bounds')
    bounds_data = get_attr_value(bounds_attr)
    if bounds_data is not None:
        if isinstance(bounds_data, (list, tuple, np.ndarray)) and len(bounds_data) >= 4:
            metadata['bounds'] = tuple(bounds_data[:4])  # (minx, miny, maxx, maxy)
    
    # Extract any other attributes (exclude the ones we've already processed)
    processed_keys = set()
    if isinstance(transform_attr, str):
        processed_keys.add(transform_attr)
    else:
        processed_keys.update(transform_attr)
    
    if isinstance(crs_attr, str):
        processed_keys.add(crs_attr)
    else:
        processed_keys.update(crs_attr)
    
    if isinstance(nodata_attr, str):
        processed_keys.add(nodata_attr)
    else:
        processed_keys.update(nodata_attr)
    
    if isinstance(bounds_attr, str):
        processed_keys.add(bounds_attr)
    else:
        processed_keys.update(bounds_attr)
    
    for key in attrs.keys():
        if key not in processed_keys:
            try:
                value = attrs[key]
                # Convert numpy types to Python types
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = value.item()
                    else:
                        value = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                elif isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8')
                    except:
                        pass
                metadata[key] = value
            except:
                pass
    
    return metadata


def create_resize_transform(config: Dict[str, Any]) -> Optional[A.Compose]:
    """
    Create albumentations resize transform from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Albumentations Compose object with resize transform, or None if resize is disabled
    """
    resize_cfg = config.get('detection', {}).get('resize', {})
    if not resize_cfg.get('enabled', False):
        return None
    
    width = resize_cfg.get('width')
    height = resize_cfg.get('height')
    interpolation = resize_cfg.get('interpolation', 'bilinear')
    
    # Map interpolation string to albumentations constant
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    interp = interpolation_map.get(interpolation.lower(), cv2.INTER_LINEAR)
    
    # Create resize transform
    if width is not None and height is not None:
        # Fixed size resize
        transform = A.Resize(height=height, width=width, interpolation=interp, always_apply=True)
    elif height is not None:
        # Resize by height, maintain aspect ratio
        transform = A.Resize(height=height, interpolation=interp, always_apply=True)
    elif width is not None:
        # Resize by width, maintain aspect ratio
        transform = A.Resize(width=width, interpolation=interp, always_apply=True)
    else:
        # No valid resize parameters
        return None
    
    return A.Compose([transform])


def predict_h5(
    h5_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    model_variant: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    imgsz: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    verbose: Optional[bool] = None
) -> List[Path]:
    """
    Predict objects in all array datasets found in an H5 file.
    
    Args:
        h5_path: Path to the H5 file (if None, uses paths.input_path from config)
        config: Configuration dictionary (if None, loads from conf_h5.yaml)
        model_path: Path to local model file (overrides config if provided)
        model_variant: Model variant to use (overrides config if provided)
        conf_threshold: Confidence threshold (overrides config if provided)
        iou_threshold: IoU threshold (overrides config if provided)
        imgsz: Input image size (overrides config if provided)
        output_dir: Directory to save output TIF files (overrides config if provided)
        verbose: Print detailed progress information (overrides config if provided)
        
    Returns:
        List of paths to saved output TIF files
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Use provided values or fall back to config
    # Get input path from argument or config
    if h5_path is None:
        h5_path = config['paths']['input_path']
        if h5_path is None:
            raise ValueError("H5 file path must be provided either via command-line argument or in config file (paths.input_path)")
    
    model_path = model_path or config['model']['local_path']
    model_variant = model_variant if model_variant is not None else config['model']['model_variant']
    conf_threshold = conf_threshold if conf_threshold is not None else config['detection']['confidence_threshold']
    iou_threshold = iou_threshold if iou_threshold is not None else config['detection']['iou_threshold']
    imgsz = imgsz if imgsz is not None else config['detection']['image_size']
    verbose = verbose if verbose is not None else config['processing']['verbose']
    continue_on_error = config['processing']['continue_on_error']
    save_output = config['output']['save']
    filename_pattern = config['output']['filename_pattern']
    
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = config['paths']['output_dir']
        if output_dir is None:
            output_dir = h5_path.parent
        else:
            output_dir = Path(output_dir)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    loader = ModelLoader()
    if model_path:
        model = loader.load_model(local_path=model_path)
    else:
        force_download = config['model']['force_download']
        model = loader.load_model(model_variant=model_variant, force_download=force_download)
    
    # Initialize detector
    detector = Detector(
        model=model,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    # Load SAM model if enabled
    sam_model = None
    sam_config = config.get('sam', {})
    sam_enabled = sam_config.get('enabled', False)
    if sam_enabled and SAM_AVAILABLE:
        sam_model_path = sam_config.get('model_path')
        if sam_model_path:
            try:
                sam_model = SAM(sam_model_path)
                if verbose:
                    print(f"Loaded SAM model: {sam_model_path}")
            except Exception as e:
                print(f"Warning: Failed to load SAM model: {e}")
                sam_enabled = False
        else:
            if verbose:
                print("Warning: SAM enabled but no model_path specified in config")
            sam_enabled = False
    elif sam_enabled and not SAM_AVAILABLE:
        print("Warning: SAM is enabled but ultralytics is not available. Install with: pip install ultralytics")
        sam_enabled = False
    
    output_files = []
    
    # Open H5 file and find all array datasets
    with h5py.File(h5_path, 'r') as h5_file:
        datasets = walk_groups(h5_file, config=config)
        
        if verbose:
            print(f"Found {len(datasets)} array dataset(s) in {h5_path.name}")
        
        for idx, (dataset_path, dataset) in enumerate(datasets, 1):
            if verbose:
                print(f"\n[{idx}/{len(datasets)}] Processing dataset: {dataset_path}")
                print("-" * 60)
            
            try:
                # Read array data
                array = np.array(dataset)
                
                if verbose:
                    print(f"  Array shape: {array.shape}")
                    print(f"  Array dtype: {array.dtype}")
                
                # Extract metadata
                metadata = extract_metadata(dataset, config)
                
                if verbose:
                    print(f"  Metadata keys: {list(metadata.keys())}")
                    if 'transform' in metadata:
                        print(f"  Transform: {metadata['transform']}")
                    if 'crs' in metadata:
                        print(f"  CRS: {metadata['crs']}")
                
                # Convert array to image format
                print("array shape: ", array.shape)
                image_array = array_to_image(array)
                print("image_array shape: ", image_array.shape)
                
                # Apply resize if configured
                resize_transform = create_resize_transform(config)
                if resize_transform is not None:
                    if verbose:
                        resize_cfg = config['detection']['resize']
                        print(f"  Resizing image: {image_array.shape[:2]} -> ", end="")
                    transformed = resize_transform(image=image_array)
                    image_array = transformed['image']
                    if verbose:
                        print(f"{image_array.shape[:2]}")
                
                # Check if tiling is needed
                tiling_cfg = config['detection'].get('tiling', {})
                use_tiling = tiling_cfg.get('enabled', False)
                tile_size = tiling_cfg.get('tile_size') or imgsz
                overlap = tiling_cfg.get('overlap', 0)
                merge_nms_iou = tiling_cfg.get('merge_nms_iou', 0.45)
                
                h, w = image_array.shape[:2]
                needs_tiling = use_tiling and (h > tile_size or w > tile_size)
                
                if needs_tiling:
                    # Use TileProcessor for tiling
                    tile_processor = TileProcessor(
                        detector=detector,
                        tile_size=tile_size,
                        overlap=overlap,
                        merge_nms_iou=merge_nms_iou,
                        verbose=verbose
                    )
                    result = tile_processor.process_image(image_array, imgsz=imgsz)
                else:
                    # Normal single-image prediction
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                    
                    try:
                        # Save image temporarily
                        cv2.imwrite(str(tmp_path), image_array)
                        
                        # Run detection
                        result = detector.detect(
                            image_path=tmp_path,
                            imgsz=imgsz,
                            save=False,
                        )
                    finally:
                        # Clean up temporary file
                        if tmp_path.exists():
                            tmp_path.unlink()
                
                # Process results (for both tiling and non-tiling)
                if result:
                    # Extract masks and generate points for SAM from detection result
                    if hasattr(result, 'masks') and result.masks is not None:
                        if verbose:
                            print("  Extracting masks and generating points for SAM...")
                        try:
                            sam_data = extract_masks_and_points(
                                result,
                                num_points_per_mask=None,  # Auto-determine based on mask area
                                min_points=1,
                                max_points=5,
                                strategy="uniform"
                            )
                            if verbose:
                                # Count actual masks (excluding fused entry)
                                num_masks = sum(1 for item in sam_data if 'mask' in item and item.get('mask_index', -1) >= 0)
                                print(f"    Generated SAM points for {num_masks} mask(s)")
                                # Find fused entry
                                fused_entry = next((item for item in sam_data if 'fused_points' in item), None)
                                if fused_entry:
                                    print(f"      Fused: {len(fused_entry['fused_points'])} total points with YOLO class labels")
                                # Print per-mask info
                                for sam_item in sam_data:
                                    if 'mask' in sam_item and 'mask_index' in sam_item and sam_item.get('mask_index', -1) >= 0:
                                        class_label = sam_item.get('class_label', sam_item['mask_index'])
                                        print(f"      Mask {sam_item['mask_index']} (class {class_label}): {len(sam_item['points'])} points")
                            
                            # Store SAM data in result for potential future use
                            if not hasattr(result, 'sam_data'):
                                result.sam_data = sam_data
                            
                            # Run SAM inference if enabled
                            if sam_enabled and sam_model is not None and len(sam_data) > 0:
                                if verbose:
                                    print("  Running SAM inference with generated points...")
                                try:
                                    # Prepare image for SAM (use original array if configured)
                                    sam_image = None
                                    original_sam_shape = None
                                    if sam_config.get('use_original_image', True):
                                        # Convert original array to image format for SAM
                                        # array_to_image returns RGB format in uint8
                                        sam_image = array_to_image(array)
                                    else:
                                        # Use processed image_array (which is in BGR format from OpenCV)
                                        sam_image = image_array.copy()
                                        # Convert BGR to RGB for SAM
                                        if len(sam_image.shape) == 3 and sam_image.shape[2] == 3:
                                            sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
                                    
                                    # Store original shape for point scaling
                                    original_sam_shape = sam_image.shape[:2]  # (height, width)
                                    
                                    # Apply resize for SAM if configured
                                    sam_resize_cfg = sam_config.get('resize', {})
                                    sam_resize_enabled = sam_resize_cfg.get('enabled', False)
                                    resize_scale_x = 1.0
                                    resize_scale_y = 1.0
                                    
                                    if sam_resize_enabled:
                                        sam_width = sam_resize_cfg.get('width')
                                        sam_height = sam_resize_cfg.get('height')
                                        sam_interpolation = sam_resize_cfg.get('interpolation', 'bilinear')
                                        
                                        # Map interpolation string to OpenCV constant
                                        interpolation_map = {
                                            'nearest': cv2.INTER_NEAREST,
                                            'bilinear': cv2.INTER_LINEAR,
                                            'bicubic': cv2.INTER_CUBIC,
                                            'area': cv2.INTER_AREA,
                                            'lanczos': cv2.INTER_LANCZOS4
                                        }
                                        interp = interpolation_map.get(sam_interpolation.lower(), cv2.INTER_LINEAR)
                                        
                                        # Calculate target size
                                        h, w = sam_image.shape[:2]
                                        if sam_width is not None and sam_height is not None:
                                            target_w, target_h = sam_width, sam_height
                                        elif sam_height is not None:
                                            # Resize by height, maintain aspect ratio
                                            target_h = sam_height
                                            target_w = int(w * (sam_height / h))
                                        elif sam_width is not None:
                                            # Resize by width, maintain aspect ratio
                                            target_w = sam_width
                                            target_h = int(h * (sam_width / w))
                                        else:
                                            target_w, target_h = w, h
                                        
                                        # Calculate scale factors for point adjustment
                                        resize_scale_x = target_w / w
                                        resize_scale_y = target_h / h
                                        
                                        # Resize image
                                        if target_w != w or target_h != h:
                                            sam_image = cv2.resize(sam_image, (target_w, target_h), interpolation=interp)
                                            if verbose:
                                                print(f"    Resized SAM image: {original_sam_shape[::-1]} -> {target_w}x{target_h}")
                                    
                                    # Extract fused points and labels from sam_data
                                    # Find the fused entry (has 'fused_points' key)
                                    fused_points = []
                                    fused_labels = []
                                    for sam_item in sam_data:
                                        if 'fused_points' in sam_item and 'fused_labels' in sam_item:
                                            fused_points = sam_item['fused_points']
                                            fused_labels = sam_item['fused_labels']
                                            break
                                    
                                    # If no fused data found, fall back to per-mask processing
                                    print("fused_points: ", fused_points)
                                    print("fused_labels: ", fused_labels)
                                    # if len(fused_points) == 0:
                                    #     # Fallback: collect points from individual masks
                                    #     for sam_item in sam_data:
                                    #         if 'points' in sam_item and 'mask_index' in sam_item:
                                    #             points = sam_item['points']
                                    #             mask_idx = sam_item['mask_index']
                                                
                                    #             for point in points:
                                    #                 fused_points.append(point)
                                    #                 fused_labels.append(mask_idx)
                                    
                                    # Format points for SAM and run inference
                                    # Pass all fused points at once with class labels
                                    sam_results_list = []
                                    
                                    if len(fused_points) > 0:
                                        # Scale all points if image was resized
                                        if sam_resize_enabled and (resize_scale_x != 1.0 or resize_scale_y != 1.0):
                                            # Scale points to match resized image
                                            points_list = [[int(p[0] * resize_scale_x), int(p[1] * resize_scale_y)] for p in fused_points]
                                        else:
                                            # Convert points to list of [x, y] lists
                                            points_list = [[int(p[0]), int(p[1])] for p in fused_points]
                                        
                                        # Use fused_labels (class labels) as labels for SAM
                                        # Note: SAM typically expects 1 for positive points, but we're using class labels
                                        labels_list = [int(label) for label in fused_labels]
                                        print("labels_list: ", labels_list)
                                        
                                        try:
                                            # Run SAM inference with all points and class labels
                                            # Format: points=[[x1, y1], [x2, y2], ...], labels=[class0, class1, ...]
                                            if verbose:
                                                unique_classes = set(fused_labels)
                                                print(f"    Running SAM with {len(points_list)} points from {len(unique_classes)} class(es): {sorted(unique_classes)}")
                                            obj_result = sam_model(sam_image, points=points_list, labels=labels_list)
                                            sam_results_list.append(obj_result)
                                        except Exception as e:
                                            if verbose:
                                                print(f"    Warning: Failed SAM inference: {e}")
                                                import traceback
                                                traceback.print_exc()
                                    
                                    if len(sam_results_list) > 0:
                                        # Store all SAM results
                                        if not hasattr(result, 'sam_results'):
                                            result.sam_results = sam_results_list
                                        
                                        if verbose:
                                            print(f"    SAM inference completed for {len(sam_results_list)} object(s)")
                                            # Print summary of results
                                            for idx, sam_res in enumerate(sam_results_list):
                                                if hasattr(sam_res, 'masks') and sam_res.masks is not None:
                                                    num_masks = len(sam_res.masks.data) if hasattr(sam_res.masks, 'data') else 0
                                                    print(f"      Object {idx}: {num_masks} mask(s)")
                                        
                                        # Save SAM results as TIF if save_output is enabled
                                        if save_output and len(sam_results_list) > 0:
                                            # Generate output filename for SAM results TIF
                                            safe_path = dataset_path.replace('/', '_').replace('\\', '_').strip('_')
                                            if not safe_path:
                                                safe_path = f"dataset_{idx}"
                                            
                                            sam_tif_filename = filename_pattern.format(
                                                h5_stem=h5_path.stem,
                                                dataset_path=safe_path,
                                                dataset_index=idx
                                            ).replace('_detections.tif', '_sam_results.tif')
                                            
                                            sam_tif_output_path = output_dir / sam_tif_filename
                                            
                                            try:
                                                # Save SAM results without resizing - use the prediction image size directly
                                                # Pass None for both shapes to prevent resizing back to original size
                                                save_detection_as_tif(
                                                    sam_results_list,
                                                    sam_image,  # Use sam_image (resized prediction image) instead of original array
                                                    metadata,
                                                    sam_tif_output_path,
                                                    detector=None,
                                                    temp_image_path=None,
                                                    sam_image_shape=None,  # Don't resize - keep prediction size
                                                    original_image_shape=None,  # Don't resize - keep prediction size
                                                    is_sam_result=True,
                                                    verbose=verbose
                                                )
                                                if verbose:
                                                    print(f"    Saved SAM results TIF to: {sam_tif_output_path}")
                                            except Exception as e:
                                                if verbose:
                                                    print(f"    Warning: Failed to save SAM results TIF: {e}")
                                                    import traceback
                                                    traceback.print_exc()
                                    
                                except Exception as e:
                                    if verbose:
                                        print(f"    Warning: Failed to run SAM inference: {e}")
                                        import traceback
                                        traceback.print_exc()
                            
                            # Save visualization with points if save_output is enabled
                            if save_output and len(sam_data) > 0:
                                # Generate output filename for SAM visualization
                                safe_path = dataset_path.replace('/', '_').replace('\\', '_').strip('_')
                                if not safe_path:
                                    safe_path = f"dataset_{idx}"
                                
                                sam_filename = filename_pattern.format(
                                    h5_stem=h5_path.stem,
                                    dataset_path=safe_path,
                                    dataset_index=idx
                                ).replace('_detections.tif', '_sam_points.jpg')
                                
                                sam_output_path = output_dir / sam_filename
                                
                                try:
                                    # Filter out fused entry for visualization (only keep actual masks)
                                    sam_data_for_viz = [item for item in sam_data 
                                                       if 'mask' in item and item.get('mask_index', -1) >= 0]
                                    save_image_with_points(
                                        image_array,
                                        sam_data_for_viz,
                                        sam_output_path,
                                        point_radius=3,
                                        point_color=(0, 255, 0),
                                        mask_alpha=0.3,
                                        show_mask=True
                                    )
                                    if verbose:
                                        print(f"    Saved SAM visualization to: {sam_output_path}")
                                except Exception as e:
                                    if verbose:
                                        print(f"    Warning: Failed to save SAM visualization: {e}")
                        except Exception as e:
                            if verbose:
                                print(f"    Warning: Failed to extract SAM data: {e}")
                    
                    # Get summary
                    summary = detector.get_detections_summary(result)
                    if verbose:
                        print(f"  Detection Summary:")
                        print(f"    Number of objects: {summary['num_detections']}")
                        if summary['num_detections'] > 0:
                            print(f"    Average confidence: {summary['average_confidence']:.3f}")
                    
                    # Generate output filename using pattern
                    # Clean dataset path for filename
                    safe_path = dataset_path.replace('/', '_').replace('\\', '_').strip('_')
                    if not safe_path:
                        safe_path = f"dataset_{idx}"
                    
                    # Format filename using pattern
                    output_filename = filename_pattern.format(
                        h5_stem=h5_path.stem,
                        dataset_path=safe_path,
                        dataset_index=idx
                    )
                    output_path = output_dir / output_filename
                    
                    # Save as TIF with metadata if save is enabled
                    if save_output:
                        # Create a temporary image for visualization
                        temp_img_path = None
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                            temp_img_path = Path(tmp_file.name)
                            cv2.imwrite(str(temp_img_path), image_array)
                        
                        save_detection_as_tif(
                            result,
                            array,
                            metadata,
                            output_path,
                            detector,
                            temp_image_path=temp_img_path
                        )
                        
                        # Clean up temp image if created
                        if temp_img_path and temp_img_path.exists():
                            temp_img_path.unlink()
                        
                        output_files.append(output_path)
                else:
                    if verbose:
                        print("  No detections found")
            
            except Exception as e:
                print(f"  Error processing dataset {dataset_path}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                if not continue_on_error:
                    raise
                continue
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing complete! Generated {len(output_files)} output file(s).")
        print(f"{'=' * 60}")
    
    return output_files


def main():
    """Main function for command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Predict objects in H5/HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process H5 file using input path from config file
  python -m src.predict.h5
  
  # Process H5 file specified via command line (overrides config)
  python -m src.predict.h5 input.h5
  
  # Use custom config file
  python -m src.predict.h5 --config my_config.yaml
  
  # Process with custom model and output directory (overrides config)
  python -m src.predict.h5 input.h5 --model-path model.pt --output-dir results/
  
  # Process with custom detection parameters (overrides config)
  python -m src.predict.h5 input.h5 --conf 0.3 --imgsz 960
        """
    )
    parser.add_argument(
        "h5_path",
        type=str,
        nargs='?',
        default=None,
        help="Path to input H5/HDF5 file (if not provided, uses paths.input_path from config)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model file (overrides --model-variant)"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help="Model variant to use if --model-path is not provided (overrides config)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold for detections (overrides config)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold for NMS (overrides config)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        choices=[640, 960, 1280],
        help="Input image size (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output TIF files (default: same directory as H5 file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: conf_h5.yaml)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    predict_h5(
        h5_path=args.h5_path,
        config=config,
        model_path=args.model_path,
        model_variant=args.model_variant,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        verbose=False if args.quiet else None
    )


if __name__ == "__main__":
    main()

