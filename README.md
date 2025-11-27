# Airplane Detection in Satellite Imagery

A Python project for detecting airplanes in satellite imagery using YOLOv8 and YOLOv9 models trained on the HRPlanes dataset. This project is based on the [Efficient-YOLO-RS-Airplane-Detection](https://github.com/RSandAI/Efficient-YOLO-RS-Airplane-Detection) repository.

## Features

- 🛫 Detect airplanes in high-resolution satellite imagery
- 🚀 Support for multiple YOLO models (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x, YOLOv9e)
- 🔧 **MMDetection support** - Use Faster R-CNN, RTMDet, YOLOX models from [OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection)
- 📸 Process single images or batches
- 🎯 Configurable confidence and IoU thresholds
- 💾 Automatic model downloading and caching
- 📊 Detailed detection summaries and visualizations

## Installation

### Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)

### Setup

1. Install dependencies using Poetry:

```bash
poetry install
```

2. Activate the virtual environment:

```bash
poetry shell
```

## Usage

### Command Line Interface

#### Single Image Detection

```bash
poetry run python -m airplane_detection.main path/to/image.jpg --model yolov8x --imgsz 960
```

#### Batch Processing

```bash
poetry run python -m airplane_detection.main path/to/images/ --model yolov8x --imgsz 960 --output results/
```

#### Available Options

- `input`: Path to input image or directory (required)
- `--model`: Model to use (`yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`, `yolov9e`) - default: `yolov8x`
- `--imgsz`: Input image size (`640`, `960`, `1280`) - default: `960`
- `--conf`: Confidence threshold (0.0-1.0) - default: `0.25`
- `--iou`: IoU threshold for NMS (0.0-1.0) - default: `0.45`
- `--output`: Output directory for results - default: `detection_results/`
- `--local-model`: Path to local model file (overrides --model)
- `--force-download`: Force re-download of model

### Python API

#### Using YOLO Models

```python
from pathlib import Path
from airplane_detection.model_loader import ModelLoader
from airplane_detection.detector import AirplaneDetector

# Load YOLO model
loader = ModelLoader()
model = loader.load_model(model_variant="flying_objects")  # or "training", "transfer"

# Initialize detector
detector = AirplaneDetector(model, conf_threshold=0.25)

# Detect airplanes
result = detector.detect("path/to/image.jpg", imgsz=960)

# Get summary
summary = detector.get_detections_summary(result)
print(f"Detected {summary['num_detections']} airplanes")

# Visualize results
detector.visualize("path/to/image.jpg", result, "output.jpg")
```

#### Using MMDetection Models

```python
from pathlib import Path
from airplane_detection.mmdetection_loader import MMDetectionLoader
from airplane_detection.detector import AirplaneDetector

# Load MMDetection model
mmdet_loader = MMDetectionLoader()
model = mmdet_loader.load_model(
    model_variant="faster_rcnn_r50",  # Options: "faster_rcnn_r50", "rtmdet_s", "yolox_s"
    class_filter=[4],  # COCO class 4 = airplane
    device="cuda:0",  # or "cpu"
)

# Initialize detector (same interface as YOLO)
detector = AirplaneDetector(model, conf_threshold=0.25)

# Detect airplanes (same workflow as YOLO)
result = detector.detect("path/to/image.jpg", imgsz=960)
summary = detector.get_detections_summary(result)
detector.visualize("path/to/image.jpg", result, "output.jpg")
```

See `example.py` for complete examples of both YOLO and MMDetection usage.

## Models

The project supports two types of detection models:

### YOLO Models

Pre-trained models from the Efficient-YOLO-RS-Airplane-Detection repository, trained on the HRPlanes dataset. Models are automatically downloaded on first use and cached in `~/.airplane_detection/models/`.

**Available YOLO Model Variants:**
- **flying_objects**: YOLOv8m model trained on flying objects dataset
- **training**: Models from training experiments (experiment-57)
- **transfer**: Models from transfer learning experiments (experiment-62)

### MMDetection Models

Models from [OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection), pre-trained on COCO dataset with airplane class filtering.

**Available MMDetection Models:**
- **faster_rcnn_r50**: Faster R-CNN with ResNet50 backbone (accurate but slower)
- **rtmdet_s**: RTMDet-S (modern, efficient detector)
- **yolox_s**: YOLOX-S (YOLO variant in MMDetection)

MMDetection models are automatically downloaded from Hugging Face on first use.

### Model Performance

Based on the original research, YOLOv8x achieved:
- F1 Score: 0.9333
- Precision: 0.9984
- Recall: 1.0
- mAP50: 0.995

## Input Image Requirements

- Supported formats: JPG, JPEG, PNG, TIF, TIFF, BMP
- Recommended resolution: High-resolution satellite imagery (0.31m spatial resolution or better)
- Image size: The model can handle various sizes, but input is resized to the specified `imgsz` parameter

## Output

The detection results include:

1. **Annotated images**: Images with bounding boxes drawn around detected airplanes
2. **Detection summary**: Number of detections, confidence scores, and bounding box coordinates
3. **Results files**: YOLO format results saved in the output directory

## Project Structure

```
airplane-detection/
├── src/
│   └── airplane_detection/
│       ├── __init__.py
│       ├── model_loader.py          # YOLO model loading and downloading
│       ├── mmdetection_loader.py    # MMDetection model loading
│       ├── detector.py              # Detection functionality
│       └── main.py                  # CLI interface
├── example.py                       # Example scripts for YOLO and MMDetection
├── tests/
├── data/
│   ├── input/                       # Place input images here
│   └── output/                      # Detection results
├── pyproject.toml
└── README.md
```

## Citation

If you use this project or the models, please cite the original research:

```bibtex
@article{ILMAK2025111854,
  title     = {Exploring You Only Look Once v8 and v9 for efficient airplane detection in very high resolution remote sensing imagery},
  journal   = {Engineering Applications of Artificial Intelligence},
  volume    = {160},
  pages     = {111854},
  year      = {2025},
  issn      = {0952-1976},
  doi       = {10.1016/j.engappai.2025.111854},
  url       = {https://www.sciencedirect.com/science/article/pii/S0952197625018561},
  author    = {Doğu İlmak and Tolga Bakirman and Elif Sertel},
  keywords  = {Airplane detection, Deep learning, You Only Look Once, Transfer learning, Optimization}
}
```

## References

- [Efficient-YOLO-RS-Airplane-Detection Repository](https://github.com/RSandAI/Efficient-YOLO-RS-Airplane-Detection)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [MMDetection - OpenMMLab Detection Toolbox](https://github.com/open-mmlab/mmdetection)
- [HRPlanes Dataset](https://zenodo.org/)

## License

This project uses models and code based on the Efficient-YOLO-RS-Airplane-Detection repository. Please refer to the original repository for license information.



