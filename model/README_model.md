# Model Module - Detailed Documentation

This directory contains the **complete YOLOv11 model implementation** for object detection on the BDD100K dataset. It includes model architecture definitions, training pipelines, inference engines, and custom dataset loaders.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Module Overview](#module-overview)
3. [Training & Inference](#training--inference)
4. [File-by-File Detailed Explanation](#file-by-file-detailed-explanation)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Inference & Deployment](#inference--deployment)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Hyperparameter Reference](#hyperparameter-reference)

---

## Quick Start

### Prerequisites

```bash
# Ensure Python 3.10+ is installed
python3 --version

# Install required packages
pip install torch torchvision pytorch-cuda=12.1 ultralytics numpy matplotlib pillow
```

### Train Model (20 epochs, ~2-3 hours on RTX A6000)

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

python3 model/train.py \
    --data_yaml configs/bdd100k.yaml \
    --epochs 20 \
    --batch_size 16 \
    --imgsz 640 \
    --device cuda \
    --model_size m \
    --output_dir runs-model
```

**Expected Output:**
```
Epoch 1/20: Loss=2.134, Val mAP@0.5=0.245 ✓
Epoch 2/20: Loss=1.856, Val mAP@0.5=0.312 ✓
...
Epoch 20/20: Loss=0.892, Val mAP@0.5=0.5415 ✓ BEST

Training Complete!
Model saved to: runs-model/bdd100k_yolo11_20epochs_.../weights/best.pt
```

### Run Inference on Single Image

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

python3 model/inference.py \
    --model_path runs-model/bdd100k_yolo11_20epochs_.../weights/best.pt \
    --image_path data/bdd100k/images/100k/val/sample.jpg \
    --conf_threshold 0.25 \
    --output_dir inference_results

# Output: Annotated image with predictions
```

### Batch Inference on Validation Set

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

python3 model/inference.py \
    --model_path runs-model/bdd100k_yolo11_20epochs_.../weights/best.pt \
    --image_dir data/bdd100k/images/100k/val \
    --conf_threshold 0.25 \
    --max_images 100 \
    --output_dir inference_results

# Processes up to 100 images with predictions
```

### Check Training Results

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

# View training metrics
python3 -c "
import json
with open('runs-model/bdd100k_yolo11_20epochs_.../training_summary.json') as f:
    summary = json.load(f)
    print(f'Final mAP@0.5: {summary[\"best_mAP\"]:.4f}')
    print(f'Final Precision: {summary[\"best_precision\"]:.4f}')
    print(f'Final Recall: {summary[\"best_recall\"]:.4f}')
"
```

---

## Module Overview

The model module orchestrates **4 core operations**:

1. **Model Definition** → YOLOv11 architecture wrapper with BDD100K customization
2. **Training** → Full training pipeline with logging, checkpointing, and validation
3. **Inference** → Prediction engine for single/batch images with visualization
4. **Data Loading** → Custom PyTorch dataset for BDD100K format

### Key Features
- Transfer learning from COCO pre-trained weights
- Multi-scale object detection (small to large)
- Anchor-free architecture for robustness
- Real-time inference (30+ FPS on GPU)
- Automatic checkpoint saving and resuming
- TensorBoard logging for monitoring
- Batch prediction support

---

## File-by-File Detailed Explanation

### 1. `model.py` — Model Architecture Wrapper
**Purpose**: Define and configure YOLOv11 for BDD100K with architecture explanations

#### Main Class: `BDD100KYOLOv11`

```python
class BDD100KYOLOv11:
    """Wrapper class for YOLOv11 model tailored for BDD100K dataset."""
    
    def __init__(
        self,
        model_size: str = 'n',      # Model size: n, s, m, l, x
        num_classes: int = 10,       # 10 classes in BDD100K
        pretrained: bool = True,     # Load COCO pre-trained weights?
        device: str = 'cuda'         # GPU or CPU
    )
```

#### Model Sizes (Trade-off: Speed vs Accuracy):

| Size | Parameters | Speed (FPS) | Accuracy | Use Case |
|------|-----------|------------|----------|----------|
| **n** (nano) | 2.6M | 100+ | Baseline | Edge devices, real-time |
| **s** (small) | 6.2M | 80 | +2% mAP | Mobile/embedded |
| **m** (medium) | 11.2M | 40 | +4% mAP | **Production standard** |
| **l** (large) | 20.1M | 20 | +6% mAP | High-accuracy applications |
| **x** (xlarge) | 56.3M | 10 | +8% mAP | Research, ensemble |

*Current deployment uses **medium (m)*** for balanced speed/accuracy.

#### Core Methods:

| Method | Purpose | Output |
|--------|---------|--------|
| `__init__()` | Initialize model with given config | Loaded model on device |
| `get_model_info()` | Get architecture details | Dict with specs |
| `print_architecture()` | Print formatted architecture explanation | Console output |
| `predict()` | Inference on image(s) | Predictions dict |
| `save_checkpoint()` | Save model weights | .pt file |

#### Model Initialization Modes:

```python
# Mode 1: Transfer Learning (RECOMMENDED)
model = BDD100KYOLOv11(model_size='m', pretrained=True)
# Loads: yolo11m.pt (pre-trained on COCO)
# Advantage: Fast convergence, good starting point

# Mode 2: From Scratch
model = BDD100KYOLOv11(model_size='m', pretrained=False)
# Loads: yolo11m.yaml (architecture only, random weights)
# Advantage: No bias from COCO, longer training required

# Mode 3: Resume Training
model = BDD100KYOLOv11(model_size='m')
model.load_checkpoint('runs-model/train/last.pt')
# Advantage: Continue from previous training
```

#### Usage Example:
```python
from model import BDD100KYOLOv11

# Initialize model
model_wrapper = BDD100KYOLOv11(
    model_size='m',
    num_classes=10,
    pretrained=True,
    device='cuda'
)

# Get info
info = model_wrapper.get_model_info()
print(f"Parameters: {info['num_classes']} classes")
model_wrapper.print_architecture()

# Inference
results = model_wrapper.predict(
    source='data/bdd100k/images/test/image.jpg',
    conf=0.25,
    iou=0.45
)
```

---

### 2. `train.py` — Training Pipeline
**Purpose**: Complete training orchestration with logging, checkpointing, and validation

#### Main Class: `BDD100KTrainer`

```python
class BDD100KTrainer:
    """Trainer class for YOLOv11 on BDD100K dataset."""
    
    def __init__(
        self,
        model_size: str = 'm',
        data_config: str = 'configs/bdd100k.yaml',
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = 'cuda',
        workers: int = 8,
        project: str = 'runs-model/train',
        name: str = 'bdd100k_yolo11',
        pretrained: bool = True,
        resume: str = None
    )
```

#### Training Configuration Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_size` | `m` | YOLOv11 model size (n/s/m/l/x) |
| `epochs` | `50` | Number of training epochs |
| `batch_size` | `16` | Batch size per GPU |
| `img_size` | `640` | Input image size (640×640 pixels) |
| `device` | `cuda` | GPU device ID or 'cpu' |
| `workers` | `8` | DataLoader workers for parallel loading |
| `optimizer` | `AdamW` | Optimization algorithm |
| `lr0` | `0.001` | Initial learning rate |
| `lrf` | `0.01` | Final learning rate (as ratio of lr0) |
| `momentum` | `0.937` | SGD momentum / Adam beta1 |
| `weight_decay` | `0.0005` | L2 regularization coefficient |
| `warmup_epochs` | `3` | Epochs for learning rate warmup |

#### Learning Rate Schedule:
```
lr(t) = lr0 × [1 + cos(π × t / epochs)] / 2

Example with lr0=0.001, epochs=50:
Epoch 0:  lr = 0.001000  (start)
Epoch 12: lr = 0.000500  (halfway)
Epoch 25: lr = 0.000001  (near end)
Epoch 50: lr = 0.000010  (final, lrf × lr0)
```

#### Core Methods:

| Method | Purpose | Output |
|--------|---------|--------|
| `setup_model()` | Initialize model (pretrained or from scratch) | Loaded YOLO model |
| `train()` | Execute full training loop | Trained weights + logs |
| `validate()` | Run validation metrics calculation | Val metrics dict |
| `save_checkpoint()` | Save model state | .pt checkpoint |
| `plot_training_curves()` | Generate loss/metric plots | PNG files |

#### Training Workflow:

```
1. Initialize trainer with config
   ↓
2. Load model (pretrained COCO weights)
   ↓
3. Create data loaders from bdd100k.yaml
   ↓
4. For each epoch:
   ├─ Warmup learning rate (first 3 epochs)
   ├─ For each batch:
   │  ├─ Forward pass: predictions
   │  ├─ Compute loss: box + class + DFL
   │  ├─ Backward pass: gradients
   │  └─ Optimizer step: update weights
   ├─ Validate on val set
   ├─ Save checkpoint if best mAP
   └─ Log metrics to TensorBoard
   ↓
5. Save final model: best.pt + last.pt
   ↓
6. Generate training curves
```

#### Loss Functions Used:

| Loss | Formula | Purpose |
|------|---------|---------|
| **Box Loss (CIoU)** | $\text{Loss} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \frac{v^2}{\pi^2} \log(1 - v)$ | Bounding box regression (considers overlap, distance, aspect ratio) |
| **Class Loss (BCE)** | $\text{Loss} = -[y \log(p) + (1-y) \log(1-p)]$ | Classification (per-class probability) |
| **DFL Loss** | Custom distribution | Improved localization accuracy (YOLOv11 specific) |

#### Data Augmentation Applied:

```python
# During training, automatically applied:
- Mosaic:        Combines 4 images into one (learns context)
- MixUp:         Blends two images (reduces overfitting)
- HSV Shift:     Hue, Saturation, Value jitter
- Rotation:      ±10° image rotation
- Scale:         ±20% random scaling
- Flip:          50% horizontal flip
- Perspective:   Random perspective transform
- Blur:          Random image blur
```

#### Usage Example:
```python
from train import BDD100KTrainer

# Create trainer
trainer = BDD100KTrainer(
    model_size='m',
    data_config='configs/bdd100k.yaml',
    epochs=50,
    batch_size=16,
    img_size=640,
    device='cuda:0',
    pretrained=True
)

# Start training
trainer.train()

# Results saved to:
# runs-model/train/bdd100k_yolo11_{timestamp}/
# ├── weights/
# │   ├── best.pt      (highest val mAP)
# │   └── last.pt      (final epoch)
# ├── results.csv      (metrics per epoch)
# └── logs/            (TensorBoard events)
```

#### Resume Training:
```python
# Resume from checkpoint
trainer = BDD100KTrainer(resume='runs-model/train/last.pt')
trainer.train()
# Continues from epoch 21 onwards
```

---

### 3. `inference.py` — Inference Engine
**Purpose**: Run predictions on images/videos with visualization

#### Main Class: `BDD100KInference`

```python
class BDD100KInference:
    """Inference class for trained YOLOv11 model."""
    
    def __init__(
        self,
        model_path: str,           # Path to trained .pt weights
        conf_threshold: float = 0.25,  # Confidence threshold
        iou_threshold: float = 0.45,   # NMS IoU threshold
        device: str = 'cuda'           # Device for inference
    )
```

#### Class Mapping & Colors:

```python
CLASSES = [
    'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motor', 'bike', 'traffic light', 'traffic sign'
]

COLORS = {
    'person': (255, 0, 0),          # Red
    'rider': (255, 128, 0),         # Orange
    'car': (0, 255, 0),             # Green
    'truck': (0, 255, 255),         # Cyan
    'bus': (0, 128, 255),           # Light Blue
    'train': (0, 0, 255),           # Blue
    'motor': (255, 0, 255),         # Magenta
    'bike': (255, 255, 0),          # Yellow
    'traffic light': (128, 0, 255), # Purple
    'traffic sign': (255, 192, 203) # Pink
}
```

#### Core Methods:

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `predict()` | Run inference | Image/video path or array | Raw predictions |
| `predict_and_visualize()` | Inference + draw boxes | Image path | Annotated image |
| `batch_predict()` | Inference on multiple images | List of paths | List of predictions |
| `predict_video()` | Inference on video | Video path | Annotated video |

#### Thresholds Explained:

**Confidence Threshold (0.25):**
```
Only detections with confidence ≥ 0.25 are kept
Higher threshold → fewer detections, higher precision
Lower threshold → more detections, higher recall
Typical: 0.25–0.5 for autonomous driving
```

**IoU Threshold (0.45):**
```
Non-Maximum Suppression (NMS) parameter
Suppresses overlapping detections (IoU > 0.45)
Higher threshold → more detections allowed
Lower threshold → more aggressive suppression
Typical: 0.4–0.5 for object detection
```

#### Prediction Output Format:

```python
{
    'boxes': tensor([
        [100, 150, 200, 250],    # [x1, y1, x2, y2] (pixels)
        [350, 100, 500, 400],
        ...
    ]),
    'labels': tensor([2, 2, 0, ...]),          # Class IDs (0-9)
    'scores': tensor([0.95, 0.87, 0.65, ...])  # Confidence scores
}
```

#### Usage Examples:

```python
from inference import BDD100KInference

# Initialize inference engine
detector = BDD100KInference(
    model_path='runs-model/train/best.pt',
    conf_threshold=0.25,
    iou_threshold=0.45,
    device='cuda'
)

# Single image inference + visualization
detector.predict_and_visualize(
    image_path='data/test_image.jpg',
    output_path='output/annotated.jpg'
)

# Batch inference on folder
results = detector.predict(
    source='data/bdd100k/images/val/',
    save=True,
    output_dir='output/predictions'
)

# Video inference
detector.predict_video(
    video_path='data/test_video.mp4',
    output_path='output/annotated_video.mp4'
)
```

#### Visualization Features:

```
Output image includes:
├─ Bounding boxes (color-coded by class)
├─ Class labels (person, car, bus, etc.)
├─ Confidence scores (0.00–1.00)
└─ Legend (class → color mapping)
```

---

### 4. `dataset_loader.py` — Custom Dataset
**Purpose**: PyTorch Dataset implementation for BDD100K format

#### Main Class: `BDD100KDataset`

```python
class BDD100KDataset(Dataset):
    """PyTorch Dataset for BDD100K object detection."""
    
    def __init__(
        self,
        image_dir: str,              # Path to images folder
        annotation_file: str,         # Path to JSON labels
        transform=None                # Optional augmentations
    )
```

#### Dataset Features:

```python
# Loading:
dataset = BDD100KDataset(
    image_dir='data/bdd100k/images/100k/train',
    annotation_file='data/bdd100k/labels/bdd100k_labels_images_train.json',
    transform=None
)

# Accessing:
for idx in range(len(dataset)):
    image, targets = dataset[idx]
    # image: PIL Image (H×W×3)
    # targets: {
    #     'boxes': tensor([[x1,y1,x2,y2], ...]),
    #     'labels': tensor([2, 0, 8, ...]),
    #     'image_id': tensor([idx])
    # }
```

#### Supported Formats:

**Input JSON Format (BDD100K):**
```json
[
  {
    "name": "0000f77c-6257be58.jpg",
    "labels": [
      {
        "category": "car",
        "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
        "attributes": {"occluded": false, "truncated": false}
      }
    ]
  }
]
```

**Output Format (PyTorch):**
```python
image = Image(H=720, W=1280, C=3)  # PIL Image
targets = {
    'boxes': [[100, 200, 300, 400], ...],  # N×4
    'labels': [2, ...],                     # N (class IDs)
    'image_id': [0]                         # [idx]
}
```

#### Data Loading Pipeline:

```
1. Load JSON annotation file
   ↓
2. Filter valid images (those that exist on disk)
   ↓
3. For each __getitem__ request:
   ├─ Load image from disk (PIL)
   ├─ Extract bounding boxes from JSON
   ├─ Extract class labels
   ├─ Convert to tensors
   ├─ Apply optional transforms
   └─ Return (image, targets)
   ↓
4. DataLoader batches them for training
```

#### Usage Example:
```python
from dataset_loader import BDD100KDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = BDD100KDataset(
    image_dir='data/bdd100k/images/100k/train',
    annotation_file='data/bdd100k/labels/bdd100k_labels_images_train.json'
)

print(f"Dataset size: {len(dataset)}")  # 70,000 images

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8
)

# Iterate batches
for batch_idx, (images, targets) in enumerate(loader):
    print(f"Batch {batch_idx}:")
    print(f"  Images shape: {images.shape}")         # [16, 3, 720, 1280]
    print(f"  Boxes shape: {targets['boxes'].shape}") # [16, N, 4]
```

---

## Model Architecture

### YOLOv11 Complete Architecture Overview

YOLOv11 is an anchor-free, end-to-end object detection model with three main components:
1. **Backbone**: Feature extraction from input images
2. **Neck**: Multi-scale feature fusion
3. **Head**: Object detection and classification

---

### Visual Architecture Flow:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INPUT IMAGE 640×640×3                           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                    ╔════════════════════════════╗
                    ║      BACKBONE STAGE        ║
                    ║    (CSPDarknet53)          ║
                    ╚════════════╤═══════════════╝
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
         ┌──────▼──────┐ ┌───────▼────────┐ ┌────▼───────────┐
         │  P3: 80×80  │ │  P4: 40×40     │ │  P5: 20×20     │
         │  8×downsmp  │ │  16×downsmp    │ │  32×downsmp    │
         │  (64 ch)    │ │  (128 ch)      │ │  (256 ch)      │
         └──────┬──────┘ └────────┬───────┘ └────┬───────────┘
                │                 │               │
                └─────────────────┼───────────────┘
                                  │
                    ╔═════════════════════════════╗
                    ║    NECK STAGE (PANet)       ║
                    ║  Top-down + Bottom-up       ║
                    ╚═════════════╤════════════════╝
                                  │
                ┌────────────────┼────────────────┐
                │                │                │
         ┌──────▼──────┐ ┌───────▼────────┐ ┌────▼───────────┐
         │  N3: 80×80  │ │  N4: 40×40     │ │  N5: 20×20     │
         │  Enhanced   │ │  Enhanced      │ │  Enhanced      │
         │  (64 ch)    │ │  (128 ch)      │ │  (256 ch)      │
         └──────┬──────┘ └────────┬───────┘ └────┬───────────┘
                │                 │               │
                └─────────────────┼───────────────┘
                                  │
                    ╔═════════════════════════════╗
                    ║  HEAD STAGE (Detection)     ║
                    ║  Decoupled Detection Head   ║
                    ╚═════════════╤════════════════╝
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   ┌────▼──────────┐  ┌──────────▼────────┐  ┌────────────▼──┐
   │ CLASSIFICATION │  │   REGRESSION      │  │  OBJECTNESS   │
   │   BRANCH       │  │      BRANCH       │  │    BRANCH     │
   ├────────────────┤  ├───────────────────┤  ├───────────────┤
   │ 10 classes     │  │ 4 coords (x,y,w,h)│  │ 1 confidence  │
   │ per location   │  │ per location      │  │ per location  │
   └────┬───────────┘  └────────┬──────────┘  └────────┬──────┘
        │                       │                      │
        └───────────────────────┼──────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
         ┌──────▼──────┐  ┌──────────────────┐  ┌─────────────┐
         │  Class Preds │  │  BBox Deltas     │  │  Confidence │
         │  (Batch, ...)│  │  (Batch, ...)    │  │  (Batch,...)│
         │  10 channels │  │  4 channels      │  │  1 channel  │
         └──────┬───────┘  └────────┬────────┘  └──────┬──────┘
                │                   │                  │
                └───────────────────┴──────────────────┘
                                  │
            ┌─────────────────────┴──────────────────┐
            │                                        │
    ┌───────▼────────────┐                  ┌───────▼────────┐
    │  POST-PROCESSING   │                  │  NMS (Dedup)   │
    │  Decode predictions│                  │  Remove overlap│
    └───────┬────────────┘                  └───────┬────────┘
            │                                       │
            └───────────────────┬────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  FINAL DETECTIONS      │
                    ├────────────────────────┤
                    │ Boxes (N×4)            │
                    │ Labels (N) - class IDs │
                    │ Scores (N) - conf.     │
                    └────────────────────────┘
```

---

### 1. BACKBONE: CSPDarknet53 (Feature Extraction)

#### Purpose:
- Extract hierarchical features at multiple scales
- Reduce spatial dimensions while increasing channel depth
- Detect features from small (edges) to large (objects)

#### Architecture Details:

| Stage | Input | Output | Operations | Purpose |
|-------|-------|--------|------------|---------|
| **Stem** | 640×640×3 | 320×320×64 | Conv 3×3 (stride 2) | Initial feature extraction |
| **Block1** | 320×320×64 | 160×160×128 | Conv + Bottleneck (CSP) | Low-level features |
| **Block2** | 160×160×128 | 80×80×256 | Conv + Bottleneck (CSP) | **P3 - Small object** |
| **Block3** | 80×80×256 | 40×40×512 | Conv + Bottleneck (CSP) | **P4 - Medium object** |
| **Block4** | 40×40×512 | 20×20×1024 | Conv + Bottleneck (CSP) | **P5 - Large object** |

#### How It Works:

```python
# Backbone processes input image
Input: 640×640×3 image
  │
  ├─ Stem Layer
  │  └─ Conv 3×3, stride=2 → 320×320×64
  │
  ├─ Block 1 (CSPDarknet)
  │  └─ Conv 3×3, stride=2 + Bottlenecks → 160×160×128
  │
  ├─ Block 2 (CSPDarknet)  
  │  └─ Conv 3×3, stride=2 + Bottlenecks → 80×80×256
  │     ✅ P3: Small objects (8× downsampling)
  │
  ├─ Block 3 (CSPDarknet)
  │  └─ Conv 3×3, stride=2 + Bottlenecks → 40×40×512
  │     ✅ P4: Medium objects (16× downsampling)
  │
  └─ Block 4 (CSPDarknet)
     └─ Conv 3×3, stride=2 + Bottlenecks → 20×20×1024
        ✅ P5: Large objects (32× downsampling)

Output: Three feature pyramid levels (P3, P4, P5)
```

#### CSP (Cross-Stage Partial) Connection:
```
Why CSP instead of regular dense connections?
┌─────────────────────────────────────────────┐
│  Dense Connection (Inefficient):            │
│  Every layer → every subsequent layer       │
│  High computation & memory overhead         │
└─────────────────────────────────────────────┘
                     VS
┌─────────────────────────────────────────────┐
│  CSP Connection (Efficient):                │
│  Split input into 2 parts:                  │
│  - Part A: Dense connections (heavy)        │
│  - Part B: Direct concat (light)            │
│  Merge at output                            │
│  Result: 20% less computation               │
└─────────────────────────────────────────────┘
```

#### Receptive Field Growth:
```
Layer    Kernel Size   Receptive Field   Purpose
─────────────────────────────────────────────────────
Stem     3×3          3                 Capture edges
Block1   3×3 (3×)     15                Low-level features
Block2   3×3 (3×)     39                Medium features
Block3   3×3 (3×)     87                High-level features
Block4   3×3 (3×)     201               Global context

By Block4, each pixel sees 201×201 input region!
```

---

### 2. NECK: PANet (Path Aggregation Network)

#### Purpose:
- Combine features from different scales
- Allow information flow from large-scale to small-scale and back
- Create feature pyramid with fused information

#### Two-Stage Fusion Process:

**Stage 1: Top-Down (Large → Small)**

```
P5 (20×20, 1024 channels)    ← Largest receptive field, deepest features
     │
     ├─ Conv 1×1 (reduce channels: 1024→512)
     │
     ├─ Upsample 2× (nearest neighbor)
     │
     ├─ Concat with P4 (40×40, 512 channels)
     │
     ├─ Conv 3×3 (bottleneck)
     │
     └─► N4_top (40×40, 512 channels)
              │
              ├─ Conv 1×1
              ├─ Upsample 2×
              ├─ Concat with P3 (80×80, 256 channels)
              ├─ Conv 3×3
              └─► N3 (80×80, 256 channels) ← Rich small-scale features
```

**Stage 2: Bottom-Up (Small → Large)**

```
N3 (80×80, 256 channels)     ← Small objects, high detail
     │
     ├─ Conv 3×3, stride=2 (downsample)
     │
     ├─ Concat with N4_top (40×40, 512 channels)
     │
     ├─ Conv 3×3 (bottleneck)
     │
     └─► N4 (40×40, 512 channels)
              │
              ├─ Conv 3×3, stride=2 (downsample)
              ├─ Concat with P5 (20×20, 1024 channels)
              ├─ Conv 3×3
              └─► N5 (20×20, 1024 channels) ← Context-rich large objects
```

#### Why This Design?

| Direction | Flow | Purpose | Benefit |
|-----------|------|---------|---------|
| **Top-Down** | Large→Small | Semantic info flows down | Small features get context |
| **Bottom-Up** | Small→Large | Location info flows up | Large features get detail |

#### Information Exchange Example:

```
For detecting a car:
  P5 (20×20): Sees entire car context (where in image)
  P3 (80×80): Sees car wheel details (sharp edges)
  
After PANet fusion:
  N5: Car context + wheel details
  N3: Wheel details + car location context
  
Result: All scales benefit from both detail AND context!
```

---

### 3. HEAD: Decoupled Detection Head (Predictions)

#### Purpose:
- Convert multi-scale features into final object detections
- Predict 3 types of information per spatial location:
  1. **What**: Class probabilities (10 classes)
  2. **Where**: Bounding box coordinates
  3. **Confidence**: How certain is the prediction

#### Architecture:

```
For each feature map (N3, N4, N5):

N3 (80×80×256)  ─┐
N4 (40×40×512)  ─┼─► HEAD
N5 (20×20×1024) ─┘
                 │
    ┌────────────┼─────────────┐
    │            │             │
    
 ┌──▼───────────┐  ┌──▼───────────┐  ┌──▼──────────┐
 │ CLS STEM      │  │ REG STEM      │  │ OBJ STEM    │
 │ (shared)      │  │ (shared)      │  │ (shared)    │
 └──┬────────────┘  └──┬───────────┘  └──┬──────────┘
    │                  │                  │
 ┌──▼──────────────┐  ┌──▼──────────┐  ┌──▼─────────┐
 │ CLS BRANCH      │  │ REG BRANCH   │  │ OBJ BRANCH │
 │ Conv 1×1        │  │ Conv 1×1     │  │ Conv 1×1   │
 │ Output: 10 ch   │  │ Output: 4 ch │  │ Output:1ch │
 └──┬──────────────┘  └──┬──────────┘  └──┬─────────┘
    │                  │                  │
    │ Class scores     │ Bbox deltas      │ Objectness
    │ [p_person,       │ [dx, dy, dw, dh] │ score
    │  p_car,          │                  │
    │  ..., p_sign]    │                  │
    │                  │                  │
    └──────────────────┼──────────────────┘
                       │
            ┌──────────▼──────────┐
            │ Decode & Postprocess │
            │ - Convert deltas     │
            │ - Apply NMS          │
            │ - Filter by conf     │
            └──────────┬───────────┘
                       │
          ┌────────────▼────────────┐
          │ FINAL DETECTIONS        │
          ├─────────────────────────┤
          │ boxes: N×[x1,y1,x2,y2]  │
          │ labels: N×class_id      │
          │ scores: N×[0,1]         │
          └─────────────────────────┘
```

#### Decoupled Head Explanation:

**Why "Decoupled"?**
- Each prediction task (classification, regression, objectness) has its own branch
- Allows optimizing each task independently
- Previous models had all tasks in one branch (harder optimization)

**Task 1: Classification (What is it?)**
```
Input: Feature from N3/N4/N5
  ├─ Shared stem (Conv 3×3, ReLU)
  ├─ Classification stem (3× Conv 1×1)
  └─ Output branch: Conv 1×1 → 10 channels (one per class)
  
Output: [p_person, p_rider, p_car, p_truck, p_bus, 
          p_train, p_motor, p_bike, p_traffic_light, p_traffic_sign]
(Before softmax: raw logits)
```

**Task 2: Regression (Where is it?)**
```
Input: Feature from N3/N4/N5
  ├─ Shared stem (Conv 3×3, ReLU)
  ├─ Regression stem (3× Conv 1×1)
  └─ Output branch: Conv 1×1 → 4 channels
  
Output: [Δx, Δy, Δw, Δh]
(Deltas from anchor point - decoded later)

Decoding:
  x = anchor_x + Δx × stride
  y = anchor_y + Δy × stride
  w = anchor_w × exp(Δw)
  h = anchor_h × exp(Δh)
```

**Task 3: Objectness (Is there an object?)**
```
Input: Feature from N3/N4/N5
  ├─ Shared stem (Conv 3×3, ReLU)
  ├─ Objectness stem (1× Conv 1×1)
  └─ Output branch: Conv 1×1 → 1 channel
  
Output: [objectness_score]
(Before sigmoid: raw logit 0-1)

Interpretation:
  > 0.5: Object likely
  < 0.5: No object / background
```

#### Spatial Output Dimensions:

```
Feature Map  Locations  Per Location     Total Predictions
─────────────────────────────────────────────────────────
N3 (80×80)   6400       10+4+1=15       6400 × 15 = 96,000
N4 (40×40)   1600       10+4+1=15       1600 × 15 = 24,000
N5 (20×20)   400        10+4+1=15       400 × 15 = 6,000
                                        Total = 126,000
```

So the model generates **126,000 potential object detections**, then NMS filters to final ~100-200 detections per image!

---

### Loss Functions & How They Work:

#### 1. **Box Loss (CIoU - Complete IoU)**

```
Formula: Loss_box = 1 - CIoU + λ × Penalty

Where:
  CIoU = IoU - (ρ²(b,b^gt) / c²) - (v²/π²) × log(1 - v)
  
  Components:
  ├─ IoU:            Basic overlap (0-1)
  ├─ Distance term:  Penalizes far predictions
  ├─ Aspect ratio:   Penalizes wrong shape
  └─ Penalty:        Extra weight for hard cases

Example:
  Perfect prediction: CIoU=1 → Loss=0 ✅
  50% overlap:        CIoU=0.5 → Loss=0.5
  Wrong aspect:       Extra penalty applied
```

#### 2. **Classification Loss (Focal Loss variant)**

```
Formula: Loss_cls = -α × (1 - p_t)^γ × log(p_t)

Where:
  α: Class weight balancing
  γ: Focusing parameter (γ=2 is typical)
  p_t: Model confidence for correct class

Purpose: Focus on hard samples (low p_t)
  Easy sample (p_t=0.9): Loss ≈ 0.0001 (ignore)
  Hard sample (p_t=0.1): Loss ≈ 0.4 (focus here)

For BDD100K (10 classes):
  - Car: α ≈ 0.5 (common, lower weight)
  - Traffic light: α ≈ 2.0 (rare, higher weight)
```

#### 3. **DFL Loss (Distribution Focal Loss)**

```
Purpose: Better localization accuracy (YOLOv11 innovation)

Traditional: Predict single (x, y, w, h) value
DFL: Predict distribution over possible values

Example for x-coordinate:
  Instead of: x = 150.5
  Predict:   P(x=149) = 0.3, P(x=150) = 0.5, P(x=151) = 0.2
  
  Advantage: Captures uncertainty, softer gradients
  Result: 1-2% mAP improvement over standard regression
```

---

### Complete Forward Pass Example:

```python
# Input: Real image of a street scene with cars and people
image = torch.randn(1, 3, 640, 640)  # 1 image, 3 channels, 640×640

# BACKBONE
p3 = backbone_stem(image)                    # 80×80×256
p4 = backbone_block1to3(p3)                  # 40×40×512
p5 = backbone_block4(p4)                     # 20×20×1024
features_pyramid = [p3, p4, p5]

# NECK (PANet Fusion)
n3_top = fuse_down(p5, p4)                   # 40×40×512
n3 = fuse_down(n3_top, p3)                   # 80×80×256
n4 = fuse_up(n3, n3_top)                     # 40×40×512
n5 = fuse_up(n4, p5)                         # 20×20×1024
enhanced_features = [n3, n4, n5]

# HEAD
outputs = []
for feat, stride in zip(enhanced_features, [8, 16, 32]):
    cls_logits = cls_head(feat)               # [B, 10, H, W]
    bbox_deltas = reg_head(feat)              # [B, 4, H, W]
    obj_scores = obj_head(feat)               # [B, 1, H, W]
    outputs.append((cls_logits, bbox_deltas, obj_scores))

# POST-PROCESSING
detections = []
for cls_logits, bbox_deltas, obj_scores in outputs:
    # Decode bbox deltas to actual coordinates
    boxes = decode_boxes(bbox_deltas)
    
    # Apply confidence threshold
    conf_mask = obj_scores > 0.25
    
    # Get class predictions
    class_ids = torch.argmax(cls_logits, dim=1)
    class_scores = torch.max(cls_logits, dim=1)[0]
    
    # Combine objectness + class confidence
    final_scores = obj_scores * class_scores
    
    # Filter by confidence
    valid = final_scores > 0.25
    detections.extend((boxes[valid], class_ids[valid], final_scores[valid]))

# NMS (Non-Maximum Suppression)
final_boxes, final_labels, final_scores = nms(detections, iou_threshold=0.45)

# Output: Detected objects with boxes, class IDs, and confidence scores
print(f"Detections: {len(final_boxes)} objects found")
# Example output:
# - Car at (100,150,250,300) with confidence 0.95
# - Person at (350,200,400,500) with confidence 0.87
# - Traffic light at (50,50,80,150) with confidence 0.72
```

#### **Loss Functions**:

1. **Box Loss (CIoU)**: Bounding box regression
   - Formula: $\text{Loss}_{box} = 1 - \text{CIoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \frac{v^2}{\pi^2}\log(1-v)$
   - Purpose: Accurate bbox localization

2. **Class Loss (Focal Loss)**: Per-class classification  
   - Formula: $\text{Loss}_{cls} = -\alpha(1-p_t)^\gamma\log(p_t)$
   - Purpose: Balanced classification, focus on hard samples

3. **DFL Loss**: Distribution focal loss for improved localization
   - Purpose: Better boundary predictions (YOLOv11 specific)
   - Result: 1-2% mAP improvement

---

## Training Pipeline

### Training Phases:

#### **Phase 1: Warmup (Epochs 1-3)**
```
Learning rate gradually increases from 0 to lr0
Stabilizes training and prevents divergence
Batch norm running stats are stabilized
```

#### **Phase 2: Decay (Epochs 4-50)**
```
Learning rate follows cosine annealing schedule
lr(t) = lr0 × [1 + cos(π × t / epochs)] / 2
Smoothly decreases to lrf × lr0 at final epoch
```

### Typical Training Curve:

```
Epoch   Box Loss   Val Loss   mAP@0.5   Notes
─────────────────────────────────────────────
0       1.37       1.34       0.42      Initial (random init)
5       1.24       1.22       0.46      Warmup phase
10      1.20       1.20       0.50      Entering decay phase
15      1.17       1.18       0.52      Still improving
20      1.15       1.17       0.54      Near convergence
30      1.13       1.16       0.55      Minimal improvement
50      1.12       1.15       0.55      Final (convergence)
```

### Hyperparameter Tuning Guide:

| Problem | Solution | Direction |
|---------|----------|-----------|
| **Underfitting** (low train loss, low val loss) | Increase model capacity | Use 'l' or 'x' model |
| | Increase training duration | More epochs |
| | Reduce regularization | Decrease weight_decay |
| **Overfitting** (low train loss, high val loss) | Add augmentation | More aggressive transforms |
| | Increase regularization | Increase weight_decay |
| | Reduce model capacity | Use 's' or 'n' model |
| | Add dropout | (implicit in YOLO) |
| **Training instability** | Reduce learning rate | Decrease lr0 |
| | Increase warmup | More warmup_epochs |
| | Smaller batch size | Reduce batch_size |
| **Slow convergence** | Increase learning rate | Increase lr0 |
| | Reduce warmup | Fewer warmup_epochs |
| | Different optimizer | Try 'SGD' or 'RMSprop' |

---

## Inference & Deployment

### Inference Speed:

```
Model Size   GPU (RTX 3090)   CPU (i7-12700)   Mobile (RPi)
────────────────────────────────────────────────────────────
Nano (n)     150 FPS          5 FPS            0.5 FPS
Small (s)    120 FPS          3 FPS            0.2 FPS
Medium (m)   40 FPS           1 FPS            0.05 FPS
Large (l)    20 FPS           0.5 FPS          0.01 FPS
```

### Deployment Options:

```
1. Python (Development):
   model = YOLO('best.pt')
   results = model.predict(source='image.jpg')

2. Edge (TensorRT):
   # Export to TensorRT for 2-3x speedup
   model.export(format='engine')

3. Mobile (ONNX):
   # Export to ONNX for cross-platform
   model.export(format='onnx')

4. Web (ONNX.js):
   # Run in browser
   model.export(format='onnx')
   # Use ONNX.js runtime
```

---

## Training & Inference

### Complete Training Workflow

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

# Train from scratch with COCO pre-trained weights
python3 model/train.py \
    --data_yaml configs/bdd100k.yaml \
    --epochs 20 \
    --batch_size 16 \
    --imgsz 640 \
    --device cuda \
    --model_size m \
    --output_dir runs-model \
    --patience 5

# Monitor training progress in another terminal:
tensorboard --logdir runs-model

# View results
cat runs-model/bdd100k_yolo11_*/training_summary.json
```

### Resume Training from Checkpoint

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

python3 model/train.py \
    --data_yaml configs/bdd100k.yaml \
    --resume_from runs-model/bdd100k_yolo11_*/weights/last.pt \
    --epochs 50 \
    --batch_size 16

# Continues training from epoch N to epoch 50
```

### Complete Inference Workflow

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

# Single image inference
python3 model/inference.py \
    --model_path runs-model/bdd100k_yolo11_*/weights/best.pt \
    --image_path data/bdd100k/images/100k/val/0000f77c-6257be58.jpg \
    --conf_threshold 0.25 \
    --output_dir inference_results

# Batch inference on directory
python3 model/inference.py \
    --model_path runs-model/bdd100k_yolo11_*/weights/best.pt \
    --image_dir data/bdd100k/images/100k/val \
    --conf_threshold 0.25 \
    --max_images 500 \
    --output_dir inference_results

# Stream inference from camera (if available)
python3 model/inference.py \
    --model_path runs-model/bdd100k_yolo11_*/weights/best.pt \
    --source 0 \
    --conf_threshold 0.25
```

---

## Usage Instructions (Detailed)---

## Performance Benchmarks

### Current Results (20 Epochs, Image Size 640×640):

```
mAP@0.5:      0.541
mAP@0.75:     0.380
Precision:    0.719
Recall:       0.496
Inference:    40 FPS (GPU), 1 FPS (CPU)
Training:     2 hours on RTX 3090
```

### Improvement Potential:

```
Strategy                          Expected Improvement
─────────────────────────────────────────────────────
Train 50+ epochs (vs 20)          +2-3% mAP
Larger model (l vs m)             +3-4% mAP
Higher resolution (1280 vs 640)   +2-3% mAP
Data augmentation tuning          +1-2% mAP
Ensemble 2-3 models               +2-4% mAP
```

---

## Hyperparameter Reference

### Default Training Configuration:

```python
{
    # Model
    'model_size': 'm',
    'num_classes': 10,
    'pretrained': True,
    
    # Training
    'epochs': 50,
    'batch_size': 16,
    'img_size': 640,
    'device': 'cuda',
    'workers': 8,
    
    # Optimization
    'optimizer': 'AdamW',
    'lr0': 0.001,           # Initial learning rate
    'lrf': 0.01,            # Final LR (as ratio)
    'momentum': 0.937,      # Momentum/beta1
    'weight_decay': 0.0005, # L2 regularization
    'warmup_epochs': 3,     # LR warmup
    
    # Augmentation
    'mosaic': 1.0,          # Mosaic probability
    'mixup': 0.1,           # MixUp probability
    'hsv_h': 0.015,         # Hue shift
    'hsv_s': 0.7,           # Saturation shift
    'hsv_v': 0.4,           # Value shift
    'degrees': 10,          # Rotation
    'translate': 0.1,       # Translation
    'scale': 0.5,           # Scaling
    'flipud': 0.0,          # Vertical flip (rare)
    'fliplr': 0.5,          # Horizontal flip
    'perspective': 0.0,     # Perspective transform
    'blur': 0.0,            # Blur
    
    # Loss weights
    'box_weight': 7.5,      # Box loss weight
    'cls_weight': 0.5,      # Class loss weight
    'dfl_weight': 1.5,      # DFL loss weight
    
    # Validation
    'val_split': 0.1,       # Validation set size
    'conf_threshold': 0.25, # Confidence threshold
    'iou_threshold': 0.45,  # NMS IoU threshold
}
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or image size
```bash
python model/train.py --batch 8 --img_size 480
```

### Issue: Training too slow
**Solution**: Increase workers, reduce validation frequency
```bash
python model/train.py --workers 16 --val_interval 5
```

### Issue: Low accuracy
**Solution**: Train more epochs, increase augmentation
```bash
python model/train.py --epochs 100 --mosaic 1.0
```

### Issue: Inference produces no detections
**Solution**: Lower confidence threshold
```python
detector.predict(image, conf=0.1)  # Instead of 0.25
```

---

## Extension Ideas

- **Add model quantization**: INT8/FP16 for faster inference
- **Add knowledge distillation**: Smaller models with teacher guidance
- **Add multi-GPU training**: Distributed training support
- **Add focal loss**: Better handling of class imbalance
- **Add custom augmentation**: Domain-specific transforms
- **Add model ensemble**: Combine multiple checkpoints

---
