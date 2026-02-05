# Model Module - Detailed Documentation

This directory contains the **complete YOLOv11 model implementation** for object detection on the BDD100K dataset. It includes model architecture definitions, training pipelines, inference engines, and custom dataset loaders.

## Table of Contents
1. [Module Overview](#module-overview)
2. [File-by-File Detailed Explanation](#file-by-file-detailed-explanation)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Inference & Deployment](#inference--deployment)
6. [Usage Instructions](#usage-instructions)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Hyperparameter Reference](#hyperparameter-reference)

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

### Visual Architecture Diagram:

```
Input Image (640×640×3)
          ↓
    ┌─────────────┐
    │  BACKBONE   │
    │(CSPDarknet) │
    └─────────────┘
    ├─ Conv(3→64)
    ├─ Conv(64→128)
    ├─ Conv(128→256)
    ├─ Conv(256→512)
    └─ Conv(512→1024)
          ↓
    Feature Maps: P3, P4, P5
    (8x, 16x, 32x downsampled)
          ↓
    ┌─────────────┐
    │    NECK     │
    │   (PANet)   │
    └─────────────┘
    ├─ Top-down FPN
    ├─ Bottom-up PAN
    └─ Multi-scale fusion
          ↓
    Enhanced Features: N3, N4, N5
          ↓
    ┌─────────────────────────────┐
    │   HEAD (Anchor-free)        │
    ├─────────────────────────────┤
    │ Classification Branch       │ → Class probabilities (10 classes)
    │ Regression Branch           │ → Bbox coords (x, y, w, h)
    │ Objectness Branch           │ → Object confidence
    └─────────────────────────────┘
          ↓
    Predictions (boxes + scores + labels)
```

### Component Details:

#### **Backbone: CSPDarknet53**
```
Purpose:   Extract multi-scale features
Mechanism: Cross-Stage Partial connections
Benefit:   20% reduction in computation
Output:    P3, P4, P5 (pyramid features)

Layers:
  Input: 640×640×3
  └─ Conv(3→64):      640×640→320×320
     └─ Conv(64→128):   320×320→160×160
        └─ Conv(128→256): 160×160→80×80    (P3: 8x downsample)
           └─ Conv(256→512): 80×80→40×40    (P4: 16x downsample)
              └─ Conv(512→1024): 40×40→20×20 (P5: 32x downsample)
```

#### **Neck: PANet (Path Aggregation Network)**
```
Purpose:   Fuse multi-scale features
Mechanism: Top-down + Bottom-up paths

Top-down (from P5 to P3):
  P5 (20×20) ──→ upsample ──→ concat with P4 ──→ N4 (40×40)
  N4 (40×40) ──→ upsample ──→ concat with P3 ──→ N3 (80×80)

Bottom-up (from N3 to N5):
  N3 (80×80) ──→ downsample ──→ concat with N4 ──→ N4' (40×40)
  N4' (40×40) ──→ downsample ──→ concat with N5 ──→ N5' (20×20)

Result: N3, N4, N5 (enhanced multi-scale features)
```

#### **Head: Decoupled Detection Head**
```
Purpose:   Make final predictions
Mechanism: Separate branches for different tasks

For each location in feature map:
├─ Classification:  10 parallel branches (one per class)
│  └─ Output: class probabilities [p_person, p_car, ..., p_sign]
├─ Regression:      4 values (x, y, w, h)
│  └─ Output: predicted bbox coordinates
└─ Objectness:      1 value
   └─ Output: confidence that object exists

Advantage: Anchor-free (simpler than anchor-based)
           Decoupled losses (better convergence)
```

#### **Loss Functions**:

1. **Box Loss (CIoU)**: Bounding box regression
2. **Class Loss (BCE)**: Per-class classification  
3. **DFL Loss**: Distribution focal loss for improved localization

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

## Usage Instructions

### Quick Start

```bash
# 1. Train model
python model/train.py --model m --epochs 50 --batch 16

# 2. Run inference on test image
python model/inference.py --model runs-model/train/best.pt --image test.jpg

# 3. Export for deployment
python -c "from ultralytics import YOLO; YOLO('runs-model/train/best.pt').export(format='onnx')"
```

### Individual Script Usage

#### Training:
```bash
python model/train.py \
    --model m \
    --data_config configs/bdd100k.yaml \
    --epochs 50 \
    --batch_size 16 \
    --img_size 640 \
    --device cuda:0
```

#### Inference (Single Image):
```bash
python model/inference.py \
    --model runs-model/train/best.pt \
    --image data/test_image.jpg \
    --conf 0.25 \
    --output output_image.jpg
```

#### Batch Inference:
```bash
python model/inference.py \
    --model runs-model/train/best.pt \
    --source data/bdd100k/images/val/ \
    --conf 0.25 \
    --output output/
```

#### Video Inference:
```bash
python model/inference.py \
    --model runs-model/train/best.pt \
    --video test_video.mp4 \
    --output output_video.mp4
```

---

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
