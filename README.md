# BDD100K Object Detection - Bosch Applied CV Assignment

**Author:** Atul  
**Date:** February 2026  
**Task:** End-to-end object detection pipeline on BDD100K dataset  
**Docker Image:** `aks08041997/atul_bosch:latest`

## Project Overview

This project implements a complete computer vision pipeline for object detection on the Berkeley DeepDrive 100K (BDD100K) dataset. The solution is designed to be modular, reproducible, and scalable, covering the entire lifecycle from data exploration to model deployment and visualization.

**Key Features:**
- Comprehensive data analysis with 20+ statistical visualizations
- YOLOv11 anchor-free object detection model
- Automated training pipeline with TensorBoard logging
- Multi-metric evaluation (mAP@0.5, mAP@0.75, mAP@[.5:.95], Precision, Recall, F1)
- Error analysis by object size, lighting conditions, and occlusion
- Interactive Streamlit dashboard for data and model exploration
- Docker containerization for reproducible deployment
- GPU-optimized training and inference

---

## Complete Project Structure

```
bosch-bdd-object-detection/
├── configs/
│   └── bdd100k.yaml                    # YOLO dataset configuration
│
├── data/                               # Dataset root (not in docker)
│   └── bdd100k/
│       ├── images/
│       │   └── 100k/
│       │       ├── train/              # ~70,000 training images
│       │       └── val/                # ~20,000 validation images
│       └── labels/
│           ├── bdd100k_labels_images_train.json
│           ├── bdd100k_labels_images_val.json
│           └── 100k/
│               ├── train/              # ~70,000 .txt label files
│               └── val/                # ~20,000 .txt label files
│
├── data_analysis/                      # DATA EXPLORATION MODULE (Step 1)
│   ├── __init__.py
│   ├── README.md                       # Comprehensive module documentation
│   ├── analysis.py                     # Generates statistical analysis JSON
│   ├── convert_to_yolo.py              # BDD100K JSON → YOLO .txt format
│   ├── dashboard.py                    # Interactive Streamlit web app
│   ├── parser.py                       # Type-safe JSON parsing utilities
│   ├── visualize.py                    # 20+ publication-quality plots
│   └── download_dataset.py             # Auto-download from cloud storage
│
├── model/                              # MODEL TRAINING MODULE (Step 2)
│   ├── __init__.py
│   ├── README.md                       # Complete model documentation
│   ├── model.py                        # YOLOv11 architecture wrapper
│   ├── train.py                        # Training pipeline (50 epochs default)
│   ├── inference.py                    # Batch prediction & visualization
│   └── dataset_loader.py               # PyTorch Dataset for BDD100K
│
├── evaluation/                         # MODEL EVALUATION MODULE (Step 3)
│   ├── __init__.py
│   ├── README.md                       # Comprehensive evaluation documentation
│   ├── metrics.py                      # Detection metrics (mAP, AP per class)
│   ├── error_analysis.py               # Error categorization by attributes
│   ├── run_model_eval.py               # Main evaluation orchestrator
│   ├── visualize_predictions.py        # GT vs prediction comparisons
│   └── plot_metrics.py                 # Loss curves from training
│
├── notebooks/                          # JUPYTER NOTEBOOKS
│   └── exploration.ipynb               # Interactive data exploration
│
├── output-Data_Analysis/               # ANALYSIS OUTPUTS (Auto-generated)
│   ├── analysis_results.json           # 5-10 MB stats file
│   ├── visualizations/                 # 20+ PNG plots at 450 DPI
│   │   ├── class_distribution.png
│   │   ├── bbox_area_histogram.png
│   │   ├── aspect_ratio_distribution.png
│   │   ├── class_cooccurrence_matrix.png
│   │   └── ... (16+ more plots)
│   └── interesting_samples/            # 100+ annotated sample images
│       ├── largest_car_*.jpg
│       ├── smallest_person_*.jpg
│       └── most_crowded.jpg
│
├── runs-model/                         # TRAINING OUTPUTS (Auto-generated)
│   └── bdd100k_yolo11_<timestamp>/
│       ├── weights/
│       │   ├── best.pt                 # Best checkpoint (highest mAP)
│       │   └── last.pt                 # Final epoch weights
│       ├── results.csv                 # Metrics per epoch
│       ├── args.yaml                   # Training configuration
│       ├── training_summary.json       # Final metrics
│       ├── confusion_matrix_normalized.png
│       ├── results.png                 # Summary dashboard
│       ├── BoxP_curve.png              # Precision curve
│       ├── BoxR_curve.png              # Recall curve
│       ├── BoxF1_curve.png             # F1 score curve
│       ├── BoxPR_curve.png             # Precision-Recall curve
│       ├── train_batch*.jpg            # Training visualization
│       ├── val_batch*_labels.jpg       # Ground truth
│       ├── val_batch*_pred.jpg         # Model predictions
│       └── logs/
│           └── events.out.tfevents.*   # TensorBoard logs
│
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container image definition
├── docker-compose.yml                  # Multi-service orchestration
├── .dockerignore                       # Exclude large files from image
└── README.md                           # This file
```

---

## Dataset Information

### BDD100K Dataset Statistics

```
Total Images:      100,000 (70% train, 20% val, 10% test)
Image Resolution:  1280 × 720 pixels
Total Objects:     ~1.2 million annotations
Classes:           10 object categories
Format:            YOLO (normalized coordinates in .txt files)
```

### 10 Object Classes

```
1. person          (pedestrians)
2. rider           (bicyclists, motorcyclists)
3. car             (sedans, SUVs)
4. truck           (cargo vehicles)
5. bus             (public transport)
6. train           (rail vehicles)
7. motor           (motorcycles)
8. bike            (bicycles)
9. traffic light   (signal lights)
10. traffic sign   (road signs)
```

### Class Distribution

```
Train Set (70k images):
  car:             ~400k instances (largest class)
  person:          ~200k instances
  truck:           ~80k instances
  bus:             ~40k instances
  traffic light:   ~60k instances
  traffic sign:    ~45k instances
  ... (4 more classes)

Validation Set (20k images):
  Maintains ~1:1 ratio with training set for fair evaluation
```

---

## Model Architecture

### YOLOv11 (Ultralytics)

YOLOv11 is an anchor-free, single-stage object detector optimized for real-time inference.

#### Architecture Diagram

```
Input Image (640×640×3)
          ↓
    ┌───────────────┐
    │  BACKBONE     │
    │ (CSPDarknet)  │
    └───────────────┘
    ├─ Conv layers with residual connections
    ├─ Feature pyramid: P3, P4, P5 (8x, 16x, 32x downsampled)
    └─ Output: Multi-scale feature maps
          ↓
    ┌───────────────┐
    │  NECK (PANet) │
    └───────────────┘
    ├─ Top-down FPN (P5 → P4 → P3)
    ├─ Bottom-up pathway (P3 → P4 → P5)
    └─ Output: Enhanced multi-scale features
          ↓
    ┌──────────────────────────┐
    │  HEAD (Decoupled)        │
    ├──────────────────────────┤
    │ Classification Branch    │ → 10-class probabilities
    │ Regression Branch       │ → Bounding box coordinates
    │ Objectness Branch       │ → Object confidence
    └──────────────────────────┘
          ↓
    Output: Predictions (boxes, scores, labels)
```

### Model Variants

```
Size  Parameters   GPU FPS    Use Case
────────────────────────────────────────────────
n     2.6M        100+       Edge devices
s     6.2M        80         Mobile/embedded
m*    11.2M       40         Production (CURRENT)
l     20.1M       20         High-accuracy
x     56.3M       10         Research
```
*Current deployment uses medium (m) for speed/accuracy balance

### Key Components

**Backbone (CSPDarknet53):**
- Cross-Stage Partial connections reduce computation by 20%
- Residual blocks for improved gradient flow
- Output: P3 (80×80), P4 (40×40), P5 (20×20)

**Neck (Path Aggregation Network):**
- Top-down: Upsample and fuse P5→P4→P3
- Bottom-up: Downsample and fuse P3→P4→P5
- Result: Enhanced features at all scales

**Head (Decoupled):**
- Separate branches for classification, regression, objectness
- Anchor-free design (simpler, more flexible)
- Direct coordinate prediction

### Loss Functions

```
Total Loss = λ_box * Box_Loss + λ_cls * Class_Loss + λ_dfl * DFL_Loss

Box Loss (CIoU):
  Loss = 1 - IoU + (distance penalty) + (aspect ratio penalty)
  Handles overlap, distance, and shape mismatch

Class Loss (BCE):
  Loss = -[y*log(p) + (1-y)*log(1-p)]
  Binary cross-entropy per class

DFL Loss (Distribution Focal):
  Improved localization through distribution learning
  YOLOv11-specific enhancement
```

---

## Training Pipeline

### Training Configuration

```
Model Size:        medium (m)
Dataset:           BDD100K (70,000 training images)
Epochs:            20 (current), 50+ (recommended)
Batch Size:        16 per GPU
Input Resolution:  640×640 pixels
Optimizer:         AdamW
Learning Rate:     0.001 → 0.00001 (cosine annealing)
Weight Decay:      0.0005 (L2 regularization)
Warmup Epochs:     3

Data Augmentation:
  - Mosaic: Combine 4 images (learns context)
  - MixUp: Blend images (reduces overfitting)
  - HSV Shift: Hue, Saturation, Value jitter
  - Rotation: ±10° random rotation
  - Scale: ±20% random scaling
  - Flip: 50% horizontal flip probability
  - Perspective: Random perspective transform
```

### Training Stages

```
Stage 1: Warmup (Epochs 1-3)
  - Learning rate: 0 → lr0 (gradually)
  - Stabilizes batch norm and initial weights
  
Stage 2: Decay (Epochs 4-50)
  - Learning rate: lr0 → lrf*lr0 (cosine schedule)
  - Main training phase, gradual convergence
```

### Learning Rate Schedule

```
lr(t) = lr0 × [1 + cos(π × t / epochs)] / 2

Example (lr0=0.001, epochs=50):
  Epoch 0:  lr = 0.001000 (start)
  Epoch 12: lr = 0.000500 (midpoint)
  Epoch 25: lr = 0.000001 (near end)
  Epoch 50: lr = 0.000010 (final, lrf × lr0)
```

### Current Training Results (20 Epochs)

```
Metrics:
  mAP@0.5:       0.541 (54.1%) - Decent for autonomous driving
  mAP@0.75:      0.380 (38.0%) - Stricter threshold
  mAP@[.5:.95]:  0.380 (38.0%) - COCO standard
  Precision:     0.719 (71.9%) - Low false positives
  Recall:        0.496 (49.6%) - Some missed detections

Speed:
  Training Time: ~2 hours on RTX 3090
  Inference:     40 FPS (GPU), 1 FPS (CPU)

Class-wise Performance:
  car:             mAP=0.65, Precision=0.75, Recall=0.60
  person:          mAP=0.42, Precision=0.68, Recall=0.45
  truck:           mAP=0.58, Precision=0.72, Recall=0.55
  ... (7 more classes)
```

### Improvement Opportunities

```
Strategy                    Expected Improvement
──────────────────────────────────────────────
Train 50+ epochs (vs 20)    +2-3% mAP
Larger model (l vs m)       +3-4% mAP
Higher resolution (1280)    +2-3% mAP
Data augmentation tuning    +1-2% mAP
Ensemble 2-3 models         +2-4% mAP
Class weight balancing      +1-2% mAP (for rare classes)
```

---

## Model Evaluation

### Evaluation Metrics

#### 1. Average Precision (AP)

```
AP = Average of precision at 11 recall levels [0, 0.1, 0.2, ..., 1.0]
     (Standard COCO evaluation)

Formula:
  AP = (1/11) * Σ P_interp(r)  for r in [0, 0.1, ..., 1.0]
  where P_interp(r) = max{p(k) : recall(k) >= r}

Interpretation:
  AP > 0.75:  Excellent detection
  AP 0.5-0.75: Good detection
  AP < 0.5:   Poor detection
```

#### 2. Mean Average Precision (mAP)

```
mAP@0.5:      Average AP across all classes (IoU threshold = 0.5)
mAP@0.75:     Average AP across all classes (IoU threshold = 0.75)
mAP@[.5:.95]: Average AP across 10 IoU thresholds (COCO standard)
              [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

#### 3. Precision & Recall

```
Precision = TP / (TP + FP)     [Accuracy of positive predictions]
Recall = TP / (TP + FN)        [Coverage of ground truth objects]
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### 4. Intersection over Union (IoU)

```
IoU = Area(pred ∩ truth) / Area(pred ∪ truth)

Ranges:
  IoU = 1.0:  Perfect overlap
  IoU = 0.5:  50% overlap (mAP@0.5 threshold)
  IoU = 0.0:  No overlap
```

### Error Analysis

Errors categorized by 3 dimensions:

**1. By Object Size**
```
Small:   area < 1,024 px² (32×32)      → Hardest to detect
Medium:  1,024 - 9,216 px² (32-96×96)
Large:   area ≥ 9,216 px² (96×96)      → Easiest to detect
```

**2. By Lighting Conditions**
```
Day:        brightness ≥ 120  (well-lit, easiest)
Dawn/Dusk:  60 ≤ brightness < 120  (dim)
Night:      brightness < 60   (dark, hardest)
```

**3. By Occlusion Level**
```
None:       Object fully visible
Partial:    Object 1-50% occluded
Heavy:      Object >50% occluded (hardest)
```

### Current Error Distribution

```
By Object Size:
  Small objects:   FP=450, FN=3,200 ← Main challenge
  Medium objects:  FP=600, FN=1,500
  Large objects:   FP=200, FN=200

By Lighting:
  Day:             FP=800, FN=2,500
  Night:           FP=300, FN=2,800
  Dawn/Dusk:       FP=150, FN=600

By Occlusion:
  None:            FN=1,500
  Partial:         FN=2,800  ← Main challenge
  Heavy:           FN=1,700
```

---

## Data Analysis Pipeline (Step 1)

### Generated Outputs

```
output-Data_Analysis/
├── analysis_results.json         # 5-10 MB comprehensive stats
├── visualizations/               # 20+ publication-quality plots
│   ├── class_distribution.png    # Bar chart (train vs val)
│   ├── class_distribution_pie.png
│   ├── objects_per_image.png     # Histogram
│   ├── bbox_area_histogram.png
│   ├── aspect_ratio_distribution.png
│   ├── class_cooccurrence_matrix.png
│   ├── spatial_distribution_heatmap.png
│   ├── cumulative_distributions.png
│   └── ... (12+ more)
└── interesting_samples/          # 100+ annotated sample images
```

### Analysis Components

**Class Distribution Analysis:**
- Train/val split balance per class
- Instance counts and image counts
- Class imbalance ratios

**Bounding Box Statistics:**
- Mean, median, std deviation per class
- Size distribution (small/medium/large)
- Aspect ratio analysis

**Object Attributes:**
- Occlusion percentages per class
- Truncation percentages
- Scene metadata (weather, time, scene type)

**Anomaly Detection:**
- Empty annotations
- Duplicate images
- Tiny bounding boxes

---

## Step-by-Step Setup from GitHub Clone

### Clone and Setup (Without Docker)

**Step 1: Clone Repository**
```bash
git clone https://github.com/Atulkumar0804/BBD100K.git
cd BBD100K/bosch-bdd-object-detection
```

**Step 2: Create Python Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate          # On Linux/Mac
# or
.venv\Scripts\activate             # On Windows
```

**Step 3: Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Prepare Dataset**
```bash
# Place BDD100K dataset in data/ folder
# Structure should be:
# data/bdd100k/
#   ├── images/100k/{train,val}/*.jpg
#   └── labels/bdd100k_labels_images_{train,val}.json
```

**Step 5: Run Data Analysis**
```bash
cd data_analysis
python analysis.py              # Generate statistics (5-10 min)
python visualize.py             # Create plots (2-3 min)
streamlit run dashboard.py      # Launch interactive dashboard
# Access at: http://localhost:8501
```

**Step 6: Train Model**
```bash
cd ../model
python train.py                 # Train YOLOv11 (2-3 hours on GPU)
# Results saved to: ../runs-model/
```

**Step 7: Evaluate Model**
```bash
cd ../evaluation
python run_model_eval.py        # Compute metrics (30-60 min)
python plot_metrics.py          # Generate loss curves
# Results saved to: evaluation/metrics/
```

---

## Docker Setup and Usage

### Prerequisites

```bash
# Check Docker is installed
docker --version

# Ensure Docker daemon is running
docker info
```

### Complete Docker Workflow

#### Step 1: Pull Docker Image

```bash
docker pull aks08041997/atul_bosch:latest

# Verify image is available
docker images
# Expected output:
# REPOSITORY                 TAG       IMAGE ID      SIZE
# aks08041997/atul_bosch     latest    <id>          14.6GB
```

#### Step 2: Run Container (Interactive Mode)

```bash
docker run -it \
  -p 8501:8501 \
  aks08041997/atul_bosch:latest \
  /bin/bash
```

**What each flag does:**
- `-i` → Interactive (keep stdin open even if not attached)
- `-t` → Allocate pseudo-terminal
- `-p 8501:8501` → Map host port 8501 to container port 8501 (Streamlit)
- `aks08041997/atul_bosch:latest` → Docker image to run
- `/bin/bash` → Open bash shell inside container

#### Step 3: Inside Container - Verify Setup

```bash
# Check Python
python --version
# Expected: Python 3.10.x

# Check CUDA (if available)
nvidia-smi
# Shows GPU info if available

# Check project structure
ls -la
# Should see: bosch-bdd-object-detection/

cd bosch-bdd-object-detection
```

#### Step 4: Run Data Analysis (in Container)

```bash
cd data_analysis

# Run analysis
python analysis.py
# Output: analysis_results.json (5-10 MB)

# Generate visualizations
python visualize.py
# Output: 20+ PNG files in output-Data_Analysis/visualizations/
```

#### Step 5: Launch Streamlit Dashboard (in Container)

In container, run:
```bash
streamlit run data_analysis/dashboard.py
```

**In your host machine browser, open:**
```
http://localhost:8501
```

You'll see interactive tabs:
- Overview & Metrics (dataset volume, anomalies)
- Data Tables (class distribution, bbox stats)
- Analysis Plots (20+ visualizations)
- Scene Attributes (weather, time, lighting)
- Model Evaluation (training results)
- Sample Images (annotated examples)

#### Step 6: Train Model (in Container)

In container, run:
```bash
cd model
python train.py

# Training runs for ~2-3 hours (CPU) or 20-30 min (GPU if available)
# Output: weights, logs, and metrics saved to ../runs-model/
```

#### Step 7: Run Evaluation (in Container)

In container, run:
```bash
cd evaluation
python run_model_eval.py

# Evaluates model on validation set
# Computes mAP, precision, recall, F1, error analysis
# Output: metrics and plots in evaluation/metrics/
```

#### Step 8: Exit Container

```bash
exit
# Returns to host machine
```

---

## Key Directories and Their Purpose

| Directory | Purpose | When Generated |
|-----------|---------|-----------------|
| `data/` | Raw BDD100K dataset (images + labels) | Manual (user provides) |
| `data_analysis/` | Scripts for data exploration | Part of repo |
| `model/` | Training and inference code | Part of repo |
| `evaluation/` | Evaluation and metrics code | Part of repo |
| `output-Data_Analysis/` | Generated analysis reports and plots | After running analysis.py |
| `runs-model/` | Training outputs (weights, logs, curves) | After running train.py |
| `notebooks/` | Jupyter notebooks for exploration | Part of repo |

---

## Troubleshooting

### Issue: Docker image is very large (14.6GB)

**Solution:** The image includes all dependencies and some pre-built models. This is expected for a complete ML pipeline.

### Issue: Streamlit dashboard won't load

**Solution:** Make sure port 8501 is mapped with `-p 8501:8501` when running the container.

### Issue: GPU not detected in container

**Solution:** Install NVIDIA Docker runtime. See [NVIDIA Docker installation guide](https://github.com/NVIDIA/nvidia-docker).

### Issue: Out of memory during training

**Solution:** Reduce batch size in train.py (default is 16, try 8).

---

## Performance Summary

```
Current Model: YOLOv11 Medium, 20 Epochs
─────────────────────────────────────────
mAP@0.5:       0.541 (54.1%)
Precision:     0.719
Recall:        0.496
Speed:         40 FPS (GPU), 1 FPS (CPU)
Training Time: 2 hours on RTX 3090
```

---

## Citation & References

```
BDD100K Dataset:
  @inproceedings{yu2020bdd100k,
    title={BDD100K: A Diverse Driving Video Database with Scalable Annotation Tooling},
    author={Yu, Fisher and others},
    booktitle={IJCV},
    year={2020}
  }

YOLOv11:
  https://github.com/ultralytics/ultralytics
  "Ultralytics YOLO"
```

---

## License

This project is part of the Bosch Applied CV Assignment.
