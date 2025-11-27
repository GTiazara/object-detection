"""MMDetection model loader for airplane detection.

This module provides support for loading MMDetection models from the
OpenMMLab detection toolbox: https://github.com/open-mmlab/mmdetection
"""

from pathlib import Path
from typing import Optional, Any, List

from huggingface_hub import hf_hub_download

# Try to import MMDetection (optional dependency)
try:
    from mmdet.apis import init_detector, inference_detector  # type: ignore
    MMDETECTION_AVAILABLE = True
except ImportError:
    MMDETECTION_AVAILABLE = False
    init_detector = None  # type: ignore
    inference_detector = None  # type: ignore


class MMDetectionLoader:
    """Handles loading and downloading MMDetection models for object detection."""
    
    # MMDetection model configurations: (repo_id, config_file, checkpoint_file, class_filter)
    # class_filter: list of class indices to filter (None = all classes)
    # Common COCO classes: 0=person, 1=bicycle, 2=car, ..., 4=airplane
    MODEL_CONFIGS = {
        # RTMDet-Tiny (lightweight, fast detector) - recommended for quick inference
        "rtmdet_tiny": (
            "open-mmlab/mmdetection",
            "configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
            "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
            [4],  # Filter for airplane class (COCO class 4)
        ),
        # RTMDet-S (modern efficient detector)
        "rtmdet_s": (
            "open-mmlab/mmdetection",
            "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
            "rtmdet_s_8xb32-300e_coco_20220905_161602-8d4e3a7e.pth",
            [4],  # Filter for airplane class
        ),
        # Faster R-CNN with ResNet50 backbone (accurate but slower)
        "faster_rcnn_r50": (
            "open-mmlab/mmdetection",
            "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
            "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
            [4],  # Filter for airplane class (COCO class 4)
        ),
        # YOLOX (YOLO variant in MMDetection)
        "yolox_s": (
            "open-mmlab/mmdetection",
            "configs/yolox/yolox_s_8xb8-300e_coco.py",
            "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
            [4],  # Filter for airplane class
        ),
    }
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the MMDetection model loader.
        
        Args:
            models_dir: Directory to store downloaded models. Defaults to ~/.airplane_detection/models
        """
        if models_dir is None:
            models_dir = Path.home() / ".airplane_detection" / "models"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_from_hf(self, repo_id: str, filename: str, force_download: bool = False) -> Path:
        """Download the specified file from Hugging Face and return the local path."""
        print(f"Downloading '{filename}' from Hugging Face repository '{repo_id}'…")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self.models_dir),
            local_dir_use_symlinks=False,
            force_download=force_download,
        )
        print(f"File available at {local_path}")
        return Path(local_path)
    
    def _get_model_files(self, model_variant: str, force_download: bool = False) -> tuple[Path, Path]:
        """
        Get config and checkpoint file paths for a model variant.
        
        This is useful if you want to use the basic MMDetection API directly:
            from mmdet.apis import init_detector, inference_detector
            config_file, checkpoint_file = loader._get_model_files("rtmdet_tiny")
            model = init_detector(str(config_file), str(checkpoint_file), device='cpu')
            inference_detector(model, 'image.jpg')
        
        Args:
            model_variant: Model variant name
            force_download: Force re-download of files
        
        Returns:
            Tuple of (config_file_path, checkpoint_file_path)
        """
        if model_variant not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown MMDetection variant '{model_variant}'. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )
        
        repo_id, config_rel_path, checkpoint_rel_path, _ = self.MODEL_CONFIGS[model_variant]
        
        config_file = self._download_from_hf(
            repo_id, config_rel_path, force_download=force_download
        )
        checkpoint_file = self._download_from_hf(
            repo_id, checkpoint_rel_path, force_download=force_download
        )
        
        return config_file, checkpoint_file
    
    def load_model(
        self,
        model_variant: str = "rtmdet_tiny",
        *,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
        class_filter: Optional[List[int]] = None,
        force_download: bool = False,
    ) -> Any:
        """
        Load an MMDetection model for object detection.
        
        This follows the MMDetection API pattern:
            from mmdet.apis import init_detector, inference_detector
            model = init_detector(config_file, checkpoint_file, device='cpu')
            inference_detector(model, 'image.jpg')
        
        Args:
            model_variant: Selects which MMDetection model to load.
                         Options: 'rtmdet_tiny', 'rtmdet_s', 'faster_rcnn_r50', 'yolox_s'
            config_path: Path to MMDetection config file (overrides variant).
            checkpoint_path: Path to checkpoint file (overrides variant).
            device: Device to run inference on ('cuda:0', 'cpu', etc.).
            class_filter: List of class indices to filter (e.g., [4] for airplane in COCO).
                        If None, uses the default from model config.
            force_download: Redownload checkpoint even if it exists locally.
        
        Returns:
            MMDetection model object wrapped with airplane filtering capability.
        
        Example:
            >>> loader = MMDetectionLoader()
            >>> model = loader.load_model("rtmdet_tiny", class_filter=[4], device="cpu")
            >>> results = model.predict("image.jpg", conf=0.25)
        """
        if not MMDETECTION_AVAILABLE:
            raise ImportError(
                "MMDetection is not installed. Install it using mim (recommended):\n"
                "  pip install openmim\n"
                "  mim install mmdet\n"
                "\n"
                "Or install manually:\n"
                "  pip install mmdet mmcv mmengine"
            )
        
        if config_path and checkpoint_path:
            # Use provided paths directly
            config_file = Path(config_path)
            checkpoint_file = Path(checkpoint_path)
        elif model_variant in self.MODEL_CONFIGS:
            # Download from Hugging Face
            repo_id, config_rel_path, checkpoint_rel_path, default_filter = self.MODEL_CONFIGS[model_variant]
            
            # Download config file
            config_file = self._download_from_hf(
                repo_id, config_rel_path, force_download=force_download
            )
            
            # Download checkpoint
            checkpoint_file = self._download_from_hf(
                repo_id, checkpoint_rel_path, force_download=force_download
            )
            
            # Use default filter if not specified
            if class_filter is None:
                class_filter = default_filter
        else:
            raise ValueError(
                f"Unknown MMDetection variant '{model_variant}'. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        print(f"Loading MMDetection model from config: {config_file}")
        print(f"Checkpoint: {checkpoint_file}")
        
        # Initialize MMDetection model using the standard MMDetection API:
        # from mmdet.apis import init_detector, inference_detector
        # model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
        # result = inference_detector(model, 'image.jpg')
        model = init_detector(str(config_file), str(checkpoint_file), device=device)
        
        # Wrap model to add class filtering and compatible interface
        return MMDetectionWrapper(model, class_filter=class_filter, device=device)


class MMDetectionWrapper:
    """Wrapper for MMDetection models to provide a compatible interface."""
    
    def __init__(self, mmdet_model: Any, class_filter: Optional[List[int]] = None, device: str = "cuda:0"):
        """
        Initialize the wrapper.
        
        Args:
            mmdet_model: Initialized MMDetection model
            class_filter: List of class indices to filter (e.g., [4] for airplane)
            device: Device string
        """
        self.model = mmdet_model
        self.class_filter = class_filter
        self.device = device
        
        # Get class names from model
        if hasattr(mmdet_model, 'dataset_meta') and mmdet_model.dataset_meta:
            self.class_names = mmdet_model.dataset_meta.get('classes', [])
        else:
            # Default COCO classes (80 classes)
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
    
    def predict(self, source: Any, conf: float = 0.25, imgsz: int = 640, **kwargs) -> List[Any]:
        """
        Run inference compatible with Ultralytics API.
        
        Args:
            source: Image path, PIL Image, or numpy array
            conf: Confidence threshold
            imgsz: Image size (for compatibility, MMDetection handles this automatically)
            **kwargs: Additional arguments
        
        Returns:
            List of Results objects compatible with detector interface
        """
        import cv2
        import numpy as np
        from PIL import Image
        
        # Load image
        if isinstance(source, (str, Path)):
            img_path = str(source)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image from {source}")
        elif isinstance(source, Image.Image):
            img = np.array(source)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_path = None
        elif isinstance(source, np.ndarray):
            img = source.copy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Assume RGB, convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_path = None
        else:
            raise ValueError(f"Unsupported image type: {type(source)}")
        
        # Run MMDetection inference using the standard API:
        # result = inference_detector(model, img)
        result = inference_detector(self.model, img)
        
        # Handle different MMDetection output formats
        # Newer versions return DetDataSample, older versions return list of arrays
        if hasattr(result, 'pred_instances'):
            # New format: DetDataSample object
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()  # (N, 4) in xyxy format
            scores = pred_instances.scores.cpu().numpy()  # (N,)
            labels = pred_instances.labels.cpu().numpy()  # (N,)
            
            # Filter by class and confidence
            if self.class_filter is not None:
                class_mask = np.isin(labels, self.class_filter)
            else:
                class_mask = np.ones(len(labels), dtype=bool)
            
            conf_mask = scores >= conf
            keep_mask = class_mask & conf_mask
            
            filtered_bboxes = bboxes[keep_mask]
            filtered_scores = scores[keep_mask]
            filtered_labels = labels[keep_mask]
            
            # Convert to list format for compatibility
            filtered_result = []
            for class_id in range(len(self.class_names)):
                class_mask = filtered_labels == class_id
                if np.any(class_mask):
                    class_bboxes = filtered_bboxes[class_mask]
                    class_scores = filtered_scores[class_mask]
                    # Convert to [x1, y1, x2, y2, score] format
                    detections = np.column_stack([class_bboxes, class_scores])
                    filtered_result.append(detections)
                else:
                    filtered_result.append(np.empty((0, 5)))
        else:
            # Old format: list of numpy arrays, one per class
            # Each array is (N, 5) with [x1, y1, x2, y2, score]
            filtered_result = []
            for class_id, detections in enumerate(result):
                # Filter by class if specified
                if self.class_filter is not None and class_id not in self.class_filter:
                    filtered_result.append(np.empty((0, 5)))
                    continue
                
                # Filter by confidence
                if len(detections) > 0:
                    mask = detections[:, 4] >= conf
                    filtered_result.append(detections[mask])
                else:
                    filtered_result.append(detections)
        
        # Create compatible results object
        return [MMDetectionResults(filtered_result, img, self.class_names, img_path)]
    
    def __call__(self, source: Any, **kwargs) -> List[Any]:
        """
        Alias for predict() to make wrapper callable like Ultralytics models.
        
        Args:
            source: Image path, PIL Image, or numpy array
            **kwargs: Additional arguments passed to predict()
        
        Returns:
            List of Results objects
        """
        return self.predict(source, **kwargs)


class MMDetectionResults:
    """Results object compatible with Ultralytics Results interface."""
    
    def __init__(self, mmdet_result: List, img: Any, class_names: List[str], img_path: Optional[str] = None):
        """
        Initialize results from MMDetection output.
        
        Args:
            mmdet_result: MMDetection inference results (list of detection arrays)
            img: Original image (BGR format)
            class_names: List of class names
            img_path: Optional path to original image
        """
        import torch
        import numpy as np
        
        self.mmdet_result = mmdet_result
        self.orig_img = img
        self.class_names = class_names
        self.path = img_path
        
        # Collect all detections across classes
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        for class_id, detections in enumerate(mmdet_result):
            if len(detections) > 0:
                for det in detections:
                    x1, y1, x2, y2, score = det
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(score)
                    all_class_ids.append(class_id)
        
        # Create boxes object compatible with Ultralytics Results
        if len(all_boxes) > 0:
            self.boxes = MMDetectionBoxes(
                torch.tensor(all_boxes, dtype=torch.float32),
                torch.tensor(all_scores, dtype=torch.float32),
                torch.tensor(all_class_ids, dtype=torch.long)
            )
        else:
            self.boxes = MMDetectionBoxes(
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long)
            )
    
    def plot(self, **kwargs) -> Any:
        """Plot detections on image (compatible with Ultralytics Results.plot)."""
        import cv2
        img = self.orig_img.copy()
        
        if self.boxes is not None and len(self.boxes.xyxy) > 0:
            boxes_np = self.boxes.xyxy.cpu().numpy()
            scores_np = self.boxes.conf.cpu().numpy()
            labels_np = self.boxes.cls.cpu().numpy()
            
            for box, score, cls_id in zip(boxes_np, scores_np, labels_np):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else 'object'} {score:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img


class MMDetectionBoxes:
    """Boxes object compatible with Ultralytics Results.boxes interface."""
    
    def __init__(self, xyxy: Any, conf: Any, cls: Any):
        """
        Initialize boxes object.
        
        Args:
            xyxy: Tensor of shape (N, 4) with bounding boxes in xyxy format
            conf: Tensor of shape (N,) with confidence scores
            cls: Tensor of shape (N,) with class indices
        """
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
    
    def cpu(self):
        """Return CPU version of boxes."""
        return MMDetectionBoxes(
            self.xyxy.cpu(),
            self.conf.cpu(),
            self.cls.cpu()
        )

