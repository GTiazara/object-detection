# Fine-Tuning YOLOv8 Models

This module allows you to fine-tune the pre-trained YOLOv8 airplane detection models on your own dataset while maintaining the same output format.

## Dataset Format

Your dataset should be organized in YOLOv8 format:

```
your_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/  (optional)
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

### Label Format

Each label file (`.txt`) should contain one bounding box per line in the format:
```
class_id x_center y_center width height
```

All coordinates are normalized (0.0 to 1.0):
- `class_id`: Class ID (0 for airplane in single-class detection)
- `x_center`: Normalized x-coordinate of box center
- `y_center`: Normalized y-coordinate of box center
- `width`: Normalized box width
- `height`: Normalized box height

Example:
```
0 0.5 0.5 0.2 0.3
0 0.3 0.7 0.15 0.25
```

## Quick Start

### 1. Prepare Your Dataset

Organize your dataset in the YOLOv8 format as shown above.

### 2. Create Dataset YAML

Create a `dataset.yaml` file in your dataset directory:

```yaml
path: /path/to/your/dataset
train: train/images
val: val/images
nc: 1
names:
  0: airplane
```

Or use the helper function:

```python
from airplane_detection.training import FineTuner

FineTuner.create_dataset_yaml(
    dataset_dir="path/to/your/dataset",
    train_dir="train",
    val_dir="val",
    num_classes=1,
    class_names=["airplane"]
)
```

### 3. Run Training

#### Using Python Script

```python
from airplane_detection.training import FineTuner

# Initialize fine-tuner with base model
fine_tuner = FineTuner(base_model_variant="training0")

# Train the model
model = fine_tuner.train(
    dataset_yaml="path/to/your/dataset/dataset.yaml",
    epochs=100,
    imgsz=960,
    batch=16,
    device="cuda",  # or "cpu"
    project="training/runs",
    name="my_fine_tune"
)
```

#### Using Command Line

```bash
python -m airplane_detection.training.train \
    --dataset-yaml path/to/your/dataset/dataset.yaml \
    --base-model training0 \
    --epochs 100 \
    --imgsz 960 \
    --batch 16 \
    --device cuda \
    --name my_fine_tune
```

### 4. Use the Trained Model

The trained model maintains the same output format as the original model. You can use it with the existing `AirplaneDetector`:

```python
from airplane_detection.model_loader import ModelLoader
from airplane_detection.detector import AirplaneDetector

# Load your fine-tuned model
loader = ModelLoader()
model = loader.load_local_model("training/runs/my_fine_tune/weights/best.pt")

# Use it for detection (same format as before)
detector = AirplaneDetector(model, conf_threshold=0.25)
result = detector.detect("path/to/image.jpg", imgsz=960)
```

## Training Parameters

- `epochs`: Number of training epochs (default: 100)
- `imgsz`: Image size for training - 640, 960, or 1280 (default: 960)
- `batch`: Batch size (default: 16, adjust based on GPU memory)
- `device`: Device to use - "cuda", "cpu", or None for auto (default: None)
- `patience`: Early stopping patience (default: 50)
- `project`: Project directory for saving results (default: "training/runs")
- `name`: Experiment name (default: "fine_tune")

## Output Format

The fine-tuned model produces the same output format as the original model:
- `Results` object from Ultralytics with:
  - `results.boxes.xyxy`: Bounding box coordinates
  - `results.boxes.conf`: Confidence scores
  - `results.plot()`: Visualization method

This ensures compatibility with your existing detection pipeline.

## Tips

1. **Start with fewer epochs**: Try 50-100 epochs first to see if the model improves
2. **Adjust batch size**: If you get out-of-memory errors, reduce the batch size
3. **Use GPU**: Training is much faster on GPU (CUDA)
4. **Monitor training**: Check the training logs and TensorBoard output in the runs directory
5. **Validate dataset**: Make sure your labels are correctly formatted before training

## Example

See `examples/train_example.py` for a complete example.

