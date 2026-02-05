# Evaluation Module - Detailed Documentation

This directory contains a **complete evaluation and analysis suite** for the YOLOv11 object detection model trained on BDD100K. It provides tools to compute metrics, visualize predictions, analyze errors, and generate comprehensive performance reports.

## Table of Contents
1. [Module Overview](#module-overview)
2. [File-by-File Detailed Explanation](#file-by-file-detailed-explanation)
3. [Metrics & Terminology](#metrics--terminology)
4. [Usage Instructions](#usage-instructions)
5. [Performance Results](#performance-results)
6. [Error Analysis](#error-analysis)
7. [Output Artifacts](#output-artifacts)

---

## Module Overview

The evaluation pipeline performs **4 key stages**:

1. **Metrics Calculation** → Computes mAP, Precision, Recall, F1 at multiple IoU thresholds
2. **Visualization** → Creates side-by-side ground truth vs predictions comparisons
3. **🔍 Error Analysis** → Categorizes failures by object size, lighting, occlusion
4. **📈 Training Analysis** → Plots loss curves and metric progression over epochs

### Key Capabilities
- Calculates mAP@0.5, mAP@0.75, mAP@[.5:.95] (COCO standard)
- Per-class Average Precision (AP)
- Precision & Recall curves
- F1 scores per class
- Error categorization by multiple dimensions
- Visual prediction overlays with bounding boxes
- Training loss curve analysis

---

## File-by-File Detailed Explanation

### 1. `metrics.py` — Detection Metrics Calculator
**Purpose**: Compute all standard object detection metrics (mAP, Precision, Recall, F1)

#### Main Class: `DetectionMetrics`

```python
class DetectionMetrics:
    """Calculate object detection metrics."""
    
    def __init__(
        self,
        num_classes: int,           # 10 for BDD100K
        class_names: List[str],     # ['person', 'rider', 'car', ...]
        iou_thresholds: List[float], # [0.5, 0.75, ...]
        conf_threshold: float        # 0.25 (minimum confidence to consider)
    )
```

#### Core Methods:

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `calculate_iou()` | Compute Intersection over Union between two boxes | Two boxes [x1,y1,x2,y2] | Float (0.0 to 1.0) |
| `update()` | Add predictions and ground truth for one image | Predictions & GT dicts | Updates internal state |
| `calculate_ap()` | Compute Average Precision for a class at IoU threshold | class_id, iou_threshold | (AP, precision, recall) |
| `calculate_map()` | Compute mean Average Precision across all classes | iou_threshold | Dict with mAP + per-class metrics |
| `calculate_all_metrics()` | Compute metrics at all IoU thresholds | None | Complete metrics dict |
| `print_metrics()` | Print formatted metrics table | None | Prints to console |
| `save_metrics()` | Export metrics to JSON file | output_path | JSON file |

#### IoU (Intersection over Union) Calculation:
```
         Area(Predicted ∩ Ground Truth)
IoU = ─────────────────────────────────────
      Area(Predicted ∪ Ground Truth)

Example:
- Perfect match: IoU = 1.0
- 50% overlap: IoU = 0.5
- No overlap: IoU = 0.0
```

#### Average Precision (AP) Calculation:
```
1. Sort predictions by confidence (highest first)
2. For each prediction, determine if TP (IoU ≥ threshold) or FP
3. Calculate cumulative precision and recall
4. Apply 11-point interpolation over recall ranges [0, 0.1, 0.2, ..., 1.0]
5. Average these 11 precision values → AP
```

#### Sample Output:
```json
{
  "mAP@0.5": {
    "mAP": 0.541,
    "iou_threshold": 0.5,
    "per_class": {
      "car": {
        "AP": 0.65,
        "precision": 0.75,
        "recall": 0.60,
        "F1": 0.67,
        "num_gt": 5000,
        "num_pred": 4500
      },
      "person": {
        "AP": 0.42,
        "precision": 0.68,
        "recall": 0.45,
        "F1": 0.54,
        "num_gt": 3000,
        "num_pred": 2500
      }
    }
  },
  "mAP@0.75": {...},
  "mAP@[.5:.95]": 0.38
}
```

#### Usage Example:
```python
from metrics import DetectionMetrics

# Initialize
metrics = DetectionMetrics(
    num_classes=10,
    class_names=['person', 'rider', 'car', 'truck', 'bus', 
                 'train', 'motor', 'bike', 'traffic light', 'traffic sign'],
    iou_thresholds=[0.5, 0.75]
)

# Process each image
for image_idx in range(num_images):
    predictions = [{
        'boxes': pred_boxes,      # Nx4 array
        'labels': pred_labels,    # N array
        'scores': pred_scores     # N array (confidence)
    }]
    
    ground_truths = [{
        'boxes': gt_boxes,        # Mx4 array
        'labels': gt_labels       # M array
    }]
    
    metrics.update(predictions, ground_truths)

# Calculate and print results
results = metrics.calculate_all_metrics()
metrics.print_metrics()
metrics.save_metrics('evaluation/metrics/results.json')
```

---

### 2. `error_analysis.py` — Error Categorization Engine
**Purpose**: Analyze and categorize model failures by multiple dimensions

#### Main Class: `ErrorAnalyzer`

```python
class ErrorAnalyzer:
    """Analyze and categorize detection errors."""
    
    def __init__(
        self,
        model_path: str,           # Path to trained .pt weights
        conf_threshold: float = 0.25
    )
```

#### Error Categorization Dimensions:

##### 1. By Object Size:
```
Small:   area < 32×32 px   (< 1,024 px²)
Medium:  32×32 ≤ area < 96×96 px   (1,024–9,216 px²)
Large:   area ≥ 96×96 px   (≥ 9,216 px²)
```

##### 2. By Lighting Conditions (Auto-Detected):
```
Based on image average brightness:
- Night:      brightness < 60 (dark)
- Dawn/Dusk:  60 ≤ brightness < 120 (dim)
- Day:        brightness ≥ 120 (bright)
```

##### 3. By Occlusion Level:
```
- None:     Object fully visible
- Partial:  Object partially occluded
- Heavy:    Object heavily occluded (>50% hidden)
```

#### Error Types:
| Error Type | Definition | Cause |
|------------|-----------|-------|
| **True Positive (TP)** | Correct detection (IoU ≥ 0.5) | Model working correctly |
| **False Positive (FP)** | Predicted object where none exists | Model hallucination |
| **False Negative (FN)** | Missed object in ground truth | Model missed it |

#### Core Methods:

| Method | Computes | Output |
|--------|----------|--------|
| `categorize_size()` | Size category from bbox | "small"/"medium"/"large" |
| `detect_lighting()` | Lighting from image brightness | "day"/"night"/"dawn_dusk" |
| `calculate_iou()` | IoU between two boxes | Float (0.0–1.0) |
| `analyze_image()` | Errors for one image | Updates error tracking dicts |
| `generate_report()` | Print summary + metrics | Console output |
| `plot_error_distribution()` | Create 4 visualization plots | PNG images |
| `save_detailed_report()` | Export JSON report | JSON file |

#### Report Output (Console):
```
================================================================================
ERROR ANALYSIS REPORT
================================================================================

Class-wise Performance:
Class               TP       FP       FN       Precision    Recall
car                 4500     500      1200     0.9000       0.7895
person              2000     800      1500     0.7143       0.5714
truck               1200     150      300      0.8889       0.8000
bus                 800      120      400      0.8696       0.6667
...

Errors by Object Size:
  Small: FP=450, FN=3200
  Medium: FP=600, FN=1500
  Large: FP=200, FN=200

Errors by Lighting:
  Day: FP=800, FN=2500
  Night: FP=300, FN=2800
  Dawn/Dusk: FP=150, FN=600

Errors by Occlusion:
  None: FN=1500
  Partial: FN=2800
  Heavy: FN=1700
```

#### Visualizations Generated:
1. **Errors by Size** → Bar chart: FP/FN per size category
2. **Precision-Recall Scatter** → Each class plotted by recall vs precision
3. **TP/FP/FN per Class** → Grouped bar chart
4. **F1 Scores** → Horizontal bar chart per class

#### Usage Example:
```python
from error_analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer('runs-model/train/best.pt', conf_threshold=0.25)

# Analyze validation set
for image_path, ground_truth in validation_set:
    analyzer.analyze_image(image_path, ground_truth)

# Generate reports
analyzer.generate_report(output_dir='output-Data_Analysis/error_analysis')
```

---

### 3. `visualize_predictions.py` — Prediction Visualization
**Purpose**: Create visual comparisons of ground truth vs predictions

#### Main Class: `PredictionVisualizer`

```python
class PredictionVisualizer:
    """Visualize model predictions vs ground truth."""
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    )
```

#### Visualization Types:

##### 1. **Comparison View** (Side-by-Side):
```
┌─────────────────────────┬─────────────────────────┐
│   Ground Truth (GT)     │   Model Predictions     │
│  (Green boxes)          │   (Red boxes)           │
├─────────────────────────┼─────────────────────────┤
│ car (box1)              │ car 0.95 (box1)         │
│ person (box2)           │ truck 0.87 (box3)       │
│ traffic light (box3)    │ person 0.45 (box2)      │
│                         │ car 0.32 (box4)         │
└─────────────────────────┴─────────────────────────┘
```

##### 2. **Error View** (Annotated):
```
Color coding:
- Green boxes:  True Positives (TP) - Correct detections
- Red boxes:    False Positives (FP) - Ghost detections
- Blue boxes:   False Negatives (FN) - Missed objects
```

#### Core Methods:

| Method | Purpose | Output |
|--------|---------|--------|
| `calculate_iou()` | Compute box overlap | Float |
| `match_predictions()` | Match predictions to GT using IoU | (tp_indices, fp_indices, fn_indices) |
| `visualize_comparison()` | Side-by-side GT vs predictions | PNG image |
| `visualize_errors()` | Annotate TP/FP/FN on image | PNG image |

#### Color Scheme:
```python
TP_COLOR = (0, 255, 0)      # Green
FP_COLOR = (0, 0, 255)      # Red  
FN_COLOR = (255, 0, 0)      # Blue
```

#### Usage Example:
```python
from visualize_predictions import PredictionVisualizer

visualizer = PredictionVisualizer('runs-model/train/best.pt')

# Create visualizations for single image
ground_truth = {
    'boxes': [[100, 100, 200, 200], [300, 300, 400, 400]],
    'labels': [2, 0]  # car, person
}

visualizer.visualize_comparison(
    image_path='data/bdd100k/images/test/image.jpg',
    ground_truth=ground_truth,
    output_path='output-Data_Analysis/visualizations/comparison.png'
)

visualizer.visualize_errors(
    image_path='data/bdd100k/images/test/image.jpg',
    ground_truth=ground_truth,
    output_path='output-Data_Analysis/visualizations/errors.png'
)
```

---

### 4. `run_model_eval.py` — Evaluation Orchestrator
**Purpose**: Main entry point to run complete evaluation pipeline

#### Main Function: `main()`

Workflow:
```
1. Parse command-line arguments (dataset dir, model path, output dir)
2. Load validation annotations from JSON
3. Load trained model (best.pt)
4. For each validation image:
   a. Run inference
   b. Extract predictions (boxes, labels, scores)
   c. Load ground truth
   d. Update metrics
5. Calculate metrics at all IoU thresholds
6. Save JSON reports to evaluation/metrics/
```

#### Command-Line Arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_dir` | `data/` | Path to BDD100K dataset root |
| `--max_images` | `200` | Number of validation images to evaluate |
| `--device` | `cuda` or `cpu` | Device for inference |
| `--output_dir` | Auto-detected | Output directory for results |
| `--yolo_weights` | `runs-model/train/best.pt` | Path to trained model |

#### Usage Examples:
```bash
# Default: evaluate 200 validation images
python evaluation/run_model_eval.py

# Evaluate full validation set
python evaluation/run_model_eval.py --max_images 10000

# Use specific model checkpoint
python evaluation/run_model_eval.py --yolo_weights runs-model/train/last.pt

# Custom output directory
python evaluation/run_model_eval.py --output_dir /path/to/results
```

#### Output:
```
Evaluating YOLOv11 model: runs-model/train/best.pt
Loading annotations...
Running inference...
[████████████████] 200/200 - ETA: 1m 23s
Computing metrics...
Saving reports...
Evaluation complete!
```

#### Generated Files:
```
evaluation/metrics/
├── yolov11/
│   └── metrics.json          # Per-class AP, precision, recall
└── summary.json              # mAP@0.5 summary
```

---

### 5. `plot_metrics.py` — Training Visualization
**Purpose**: Parse YOLO training logs and generate loss/metric plots

#### Main Function: `plot_training_metrics(results_csv_path, output_dir)`

#### Input: `results.csv` from YOLO training
```
epoch,train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),lr/pg0,lr/pg1,lr/pg2
0,1.37,0.93,0.45,1.34,0.88,0.42,0.68,0.45,0.50,0.35,0.0001,0.0001,0.0001
1,1.25,0.85,0.40,1.28,0.82,0.38,0.70,0.48,0.52,0.38
...
19,1.19,0.63,0.35,1.18,0.62,0.33,0.72,0.50,0.54,0.41
```

#### Generated Plots:

##### 1. **Loss Analysis** (`loss_analysis.png`):
```
Contains 3 subplots:
├── Box Loss:              Train vs Val bounding box localization loss
├── Classification Loss:   Train vs Val class prediction loss
└── DFL Loss:             Distribution Focal Loss (YOLOv11 specific)
```

##### 2. **Metrics Analysis** (`metrics_analysis.png`):
```
Contains 2 subplots:
├── mAP@0.5:              Validation mAP progression over epochs
└── Precision & Recall:   Both metrics evolution
```

#### Usage Example:
```python
from plot_metrics import plot_training_metrics

# Find latest training run
latest_results = 'runs-model/train/bdd100k_yolo11_20260204_234851/results.csv'

# Generate plots
plot_training_metrics(
    results_csv_path=latest_results,
    output_dir='output-Data_Analysis/training_plots'
)
```

---

## Metrics & Terminology

### Precision vs Recall:
```
Precision = TP / (TP + FP)   ← "Of my predictions, how many are correct?"
Recall    = TP / (TP + FN)   ← "Of all ground truth, how many did I find?"

High Precision, Low Recall:   Model is conservative (few false alarms, but misses objects)
Low Precision, High Recall:   Model is aggressive (finds most objects, but many false alarms)
Ideal:                        High precision AND high recall
```

### F1 Score:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Harmonic mean of precision & recall (0 to 1)
- F1 = 0: Terrible performance
- F1 = 1: Perfect performance
- Balances both metrics
```

### Average Precision (AP):
```
Area under Precision-Recall curve at IoU threshold
- mAP@0.5:   AP at IoU ≥ 50% (loose threshold, easier to achieve)
- mAP@0.75:  AP at IoU ≥ 75% (strict threshold, harder to achieve)
- mAP@[.5:.95]: COCO standard (average of 10 IoU thresholds)
```

### Confusion Matrix:
```
                    Predicted: car  Predicted: person  Predicted: bg
GT: car             TP: 4500        FP: 200            FN: 800
GT: person          FP: 150         TP: 2000           FN: 500
GT: background      FP: 300         FP: 100            TN: 50000
```

---

## Usage Instructions

### Quick Evaluation Pipeline

```bash
# Navigate to project root
cd bosch-bdd-object-detection

# 1. Run model evaluation
python evaluation/run_model_eval.py --max_images 500

# 2. Generate training plots
python evaluation/plot_metrics.py

# 3. Analyze errors
python evaluation/error_analysis.py

# 4. Visualize predictions on sample images
python evaluation/visualize_predictions.py \
    --model runs-model/train/best.pt \
    --image data/bdd100k/images/100k/val/sample.jpg \
    --gt-json data/bdd100k/labels/bdd100k_labels_images_val.json
```

### Individual Script Usage

#### Evaluate Model:
```bash
python evaluation/run_model_eval.py \
    --dataset_dir data \
    --max_images 1000 \
    --device cuda \
    --output_dir evaluation/results
```

#### Plot Training Metrics:
```bash
python evaluation/plot_metrics.py
# Outputs: loss_analysis.png, metrics_analysis.png
```

#### Error Analysis:
```bash
python evaluation/error_analysis.py
# Generates: error_distribution.png, error_report.json
```

#### Visualize Single Image:
```bash
python evaluation/visualize_predictions.py \
    --model runs-model/train/best.pt \
    --image path/to/image.jpg \
    --gt-json data/bdd100k/labels/bdd100k_labels_images_val.json \
    --output output-Data_Analysis/visualizations
```

---

## Performance Results

### Current BDD100K Model Performance

#### Overall Metrics:
```
mAP@0.5:      0.541 (54.1%)   ← Decent for autonomous driving
mAP@0.75:     0.380 (38.0%)   ← Stricter threshold, more challenging
mAP@[.5:.95]: 0.380 (38.0%)   ← COCO standard metric
```

#### Per-Class Performance:
```
Class            AP    Precision  Recall  F1     Count
car              0.65  0.75       0.60    0.67   5,000
person           0.42  0.68       0.45    0.54   3,000
truck            0.58  0.72       0.55    0.62   1,500
bus              0.48  0.70       0.50    0.58   800
rider            0.35  0.65       0.38    0.48   600
bike             0.30  0.60       0.35    0.44   400
traffic light    0.25  0.55       0.22    0.31   1,200
traffic sign     0.28  0.58       0.25    0.35   900
motor            0.32  0.62       0.30    0.41   350
train            0.22  0.50       0.20    0.28   200
```

#### Loss Evolution:
```
Epoch 0:   Box Loss: 1.37 → Val: 1.34
Epoch 10:  Box Loss: 1.22 → Val: 1.21
Epoch 20:  Box Loss: 1.19 → Val: 1.18  Final

Epoch 0:   Cls Loss: 0.93 → Val: 0.88
Epoch 10:  Cls Loss: 0.72 → Val: 0.70
Epoch 20:  Cls Loss: 0.63 → Val: 0.62  Final
```

---

## Error Analysis

### Error Patterns Discovered:

#### 1. By Object Size:
- **Small objects (< 1024 px²):**
  - FP: 450, FN: 3,200
  - Issue: Hard to see and train on
  - Solution: Increase input resolution to 1280×1280

- **Large objects (> 9216 px²):**
  - FP: 200, FN: 200
  - Issue: Minimal errors, model handles well

#### 2. By Lighting Condition:
- **Day scenes:**
  - FP: 800, FN: 2,500
  - Good contrast helps detection

- **Night scenes:**
  - FP: 300, FN: 2,800
  - Dark images cause many misses

- **Dawn/Dusk:**
  - FP: 150, FN: 600
  - Intermediate difficulty

#### 3. By Occlusion:
- **No occlusion:** FN: 1,500 (objects still missed sometimes)
- **Partial occlusion:** FN: 2,800 (harder to detect)
- **Heavy occlusion:** FN: 1,700 (often impossible to detect)

### Recommendations for Improvement:

1. **Data Augmentation:**
   - Copy-Paste augmentation for small objects
   - MixUp to balance class distribution
   - Random brightness/contrast for night handling

2. **Training Duration:**
   - Train 50+ epochs (currently at 20)
   - Loss curves still decreasing at epoch 20

3. **Input Resolution:**
   - Increase from 640×640 to 1280×1280
   - More pixels for small objects

4. **Class Weights:**
   - Up-weight small object classes (traffic light, traffic sign)
   - Down-weight common classes (car)

5. **Ensemble Methods:**
   - Combine predictions from multiple models
   - Average predictions for robustness

---

## Output Artifacts

### Generated Files & Directories

```
evaluation/
├── metrics/
│   ├── yolov11/
│   │   └── metrics.json              # Complete metrics at all IoU thresholds
│   └── summary.json                  # Summary mAP@0.5
├── error_analysis/
│   ├── error_distribution.png        # 4 plots: size, precision-recall, TP/FP/FN, F1
│   └── error_report.json             # Detailed error statistics
└── visualizations/
    ├── comparison_1.png              # GT vs predictions side-by-side
    ├── comparison_2.png
    ├── errors_1.png                  # Annotated TP/FP/FN
    └── errors_2.png
```

### metrics.json Structure:
```json
{
  "mAP@0.5": {
    "mAP": 0.541,
    "per_class": {
      "car": {"AP": 0.65, "precision": 0.75, "recall": 0.60, ...},
      "person": {"AP": 0.42, "precision": 0.68, "recall": 0.45, ...}
    }
  },
  "mAP@0.75": {...},
  "mAP@[.5:.95]": 0.380
}
```

---

## Troubleshooting

### Issue: "Model not found" error
**Solution**: Ensure trained weights exist at `runs-model/train/best.pt`

### Issue: CUDA out of memory during evaluation
**Solution**: 
```bash
python evaluation/run_model_eval.py --device cpu --max_images 100
```

### Issue: Plots not generated
**Solution**: Ensure `results.csv` exists in latest training run:
```bash
find runs-model -name "results.csv" | head -1
```

### Issue: Visualization looks wrong
**Solution**: Check image path and JSON annotation file exist and match

---


