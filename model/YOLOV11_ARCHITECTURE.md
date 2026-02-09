# YOLOv11 Architecture - Comprehensive Guide

## Overview
YOLOv11 (2024) is the latest advancement in the YOLO (You Only Look Once) family of real-time object detectors. It represents a significant evolution in detection architecture, offering improved speed-accuracy tradeoffs through refined backbone design, efficient feature fusion, and optimized detection heads.

---

## 1. Complete Architecture Pipeline

```
Input Image (1280×720 or 640×640)
    ↓
[BACKBONE: CSPDarknet53]
Extract multi-scale features
    ↓
    ├─ Stem: 3×3 Conv (32 filters)
    ├─ Stage 1: 3×3 Conv (64 filters) + 3×C2f Blocks
    ├─ Stage 2: 3×3 Conv (128 filters) + 6×C2f Blocks
    ├─ Stage 3: 3×3 Conv (256 filters) + 6×C2f Blocks
    ├─ Stage 4: 3×3 Conv (512 filters) + 3×C2f Blocks
    └─ SPPF: Spatial Pyramid Pooling (Fast)
    ↓
Feature Maps: P3 (80×80), P4 (40×40), P5 (20×20)
    ↓
[NECK: PANet - Path Aggregation Network]
Bi-directional feature fusion
    ↓
    ├─ Top-down: P5→P4→P3 (upsampling + concatenation)
    └─ Bottom-up: P3→P4→P5 (downsampling + concatenation)
    ↓
Enhanced Features: P3', P4', P5'
    ↓
[HEAD: Decoupled Detection Head]
Three parallel branches
    ↓
    ├─ Classification Branch → cls (80 classes)
    ├─ Regression Branch → bbox (4 values: x, y, w, h)
    └─ Objectness Branch → obj (confidence score)
    ↓
Raw Predictions: 126,000 anchors × 85 values
    ↓
Post-Processing (NMS, Confidence Threshold)
    ↓
Final Detections
```

---

## 2. BACKBONE: CSPDarknet53 Architecture

### Purpose
The backbone extracts hierarchical features from the input image at multiple scales. It efficiently captures both low-level details (edges, textures) and high-level semantics (objects, scenes).

### Architecture Stages

| Stage | Input | Layer | Filters | Output | Receptive Field |
|-------|-------|-------|---------|--------|-----------------|
| **Stem** | 1280×720 | Conv 3×3, stride=2 | 32 | 640×360 | 3×3 |
| **C2f Block** | 640×360 | Conv + C2f (Bottleneck) | 32 | 640×360 | ~7×7 |
| **Stage 1** | 640×360 | Conv 3×3, stride=2 | 64 | 320×180 | ~15×15 |
| | 320×180 | 3× C2f Blocks | 64 | 320×180 | ~31×31 |
| **Stage 2** | 320×180 | Conv 3×3, stride=2 | 128 | 160×90 | ~63×63 |
| | 160×90 | 6× C2f Blocks | 128 | 160×90 | ~127×127 |
| **Stage 3** | 160×90 | Conv 3×3, stride=2 | 256 | 80×45 | ~255×255 |
| | 80×45 | 6× C2f Blocks | 256 | 80×45 | ~511×511 |
| **Stage 4** | 80×45 | Conv 3×3, stride=2 | 512 | 40×22 | ~1023×1023 |
| | 40×22 | 3× C2f Blocks | 512 | 40×22 | ~2047×2047 |
| **SPPF** | 40×22 | Spatial Pyramid Pooling | 512 | 40×22 | ~2047×2047 |

### Key Components

#### C2f Block (Bottleneck with Split Concatenation)
```python
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of Bottleneck blocks
            shortcut: Skip connection flag
            g: Groups for grouped convolution
            e: Expansion ratio (0.5 means half channels in bottleneck)
        """
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Split conv
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)  # Concatenation conv
        
        # N Bottleneck blocks
        self.m = nn.Sequential(
            *[Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))  # Split into 2 parts
        y.extend(m(y[-1]) for m in self.m)  # Pass through bottlenecks
        return self.cv2(torch.cat(y, 1))  # Concatenate and merge
```

#### SPPF (Spatial Pyramid Pooling Fast)
```python
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        """
        Args:
            c1: Input channels
            c2: Output channels
            k: Pool kernel size (default 5×5)
        """
        self.cv1 = Conv(c1, c1 // 2, 1, 1)
        self.cv2 = Conv(c1 * 2, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        # Multi-scale max pooling (1×1, 5×5, 5×5, 5×5)
        y1 = self.m(x)  # 5×5 pool
        y2 = self.m(y1)  # 5×5 pool (2 times)
        return self.cv2(torch.cat([x, y1, y2], 1))  # Concatenate 4 scales
```

**Why SPPF matters:**
- Captures features at multiple spatial scales in a single location
- Improves receptive field without stacking many conv layers
- Particularly effective for detecting objects of varying sizes
- For BDD100K (cars, pedestrians, cyclists): SPPF captures both small and large objects

### Backbone Output
- **P3** (80×80): Small objects (pedestrians ~40px, cyclists ~50px)
- **P4** (40×40): Medium objects (cars ~100px)
- **P5** (20×20): Large objects (distant cars ~200px)

---

## 3. NECK: PANet - Path Aggregation Network

### Purpose
The neck combines features from different scales to create a unified representation that captures both fine details and semantic information. It enables detection of objects at all scales.

### Two-Stage Fusion Process

#### Stage 1: Top-Down Fusion (Semantic Enhancement)
```
P5 (20×20, 512 filters)
    ↓ Conv 1×1 → 256 filters
    ↓ Upsample 2× (nearest neighbor)
    ↓ Concatenate with P4
    ↓ Conv 1×1 → 256 filters
    ↓ C2f Bottleneck Block
    → P4_enhanced (40×40, 256 filters)
    
P4_enhanced
    ↓ Conv 1×1 → 256 filters
    ↓ Upsample 2×
    ↓ Concatenate with P3
    ↓ Conv 1×1 → 128 filters
    ↓ C2f Bottleneck Block
    → P3_enhanced (80×80, 128 filters)
```

**What happens:** Semantic information from P5 flows down to enhance smaller features with context.

#### Stage 2: Bottom-Up Fusion (Spatial Enhancement)
```
P3_enhanced (80×80, 128 filters)
    ↓ Conv 3×3, stride=2 → 128 filters
    ↓ Concatenate with P4_enhanced
    ↓ Conv 1×1 → 256 filters
    ↓ C2f Bottleneck Block
    → P4_final (40×40, 256 filters)
    
P4_final
    ↓ Conv 3×3, stride=2 → 256 filters
    ↓ Concatenate with P5
    ↓ Conv 1×1 → 512 filters
    ↓ C2f Bottleneck Block
    → P5_final (20×20, 512 filters)
```

**What happens:** Spatial detail from P3 flows up to enhance larger feature maps.

### Feature Fusion Example (Car Detection)

| Feature Level | Resolution | Purpose | Example: Car Detection |
|---|---|---|---|
| **P3 (80×80)** | Fine detail | Small objects | Distant cars (20-40 pixels) |
| After Top-down | 80×80 | + Semantic context from scene | Knows it's in a road scene |
| After Bottom-up | 80×80 | + Spatial detail from small cars | Precisely localizes small distant cars |
| **P4 (40×40)** | Medium detail | Medium objects | Normal-distance cars (80-120px) |
| After Top-down | 40×40 | + Context from highway scene | Knows it's part of traffic |
| After Bottom-up | 40×40 | + Edge detail from small cars | Clear boundary detection |
| **P5 (20×20)** | Coarse detail | Large objects | Close cars (200+ pixels) |
| After Top-down | 20×20 | + Texture details | Mirrors, windows visible |
| After Bottom-up | 20×20 | + Spatial relationships | Car position relative to lane |

### Why This Design

| Benefit | Explanation |
|---------|-------------|
| **Multi-scale awareness** | Each feature map knows about all scales |
| **Better localization** | Fine details for precise bounding boxes |
| **Improved confidence** | Multiple sources of evidence for each prediction |
| **Efficient** | Reuses feature maps rather than computing separately |
| **Robust to scale variation** | BDD100K has cars from many distances |

---

## 4. HEAD: Decoupled Detection Head

### Purpose
The head makes final predictions: **Where** are objects? **What** are they? **How confident** are we?

### Architecture: Three Parallel Branches

```python
class Detect(nn.Module):
    def __init__(self, nc=80, filters=(128, 256, 512)):
        """
        Args:
            nc: Number of classes (80 for BDD100K COCO-format)
            filters: Channel sizes for P3, P4, P5
        """
        self.nc = nc
        self.nl = 3  # Three prediction levels
        self.no = nc + 5  # 80 classes + 4 bbox coords + 1 confidence = 85
        self.stride = [8, 16, 32]  # Strides for P3, P4, P5
        
        # Branch 1: Classification
        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                Conv(f, f, 3, 1),
                Conv(f, f, 3, 1),
                nn.Conv2d(f, self.nc, 1)
            ) for f in filters
        ])
        
        # Branch 2: Regression (Bounding Box)
        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                Conv(f, f, 3, 1),
                Conv(f, f, 3, 1),
                nn.Conv2d(f, 4, 1)  # x, y, w, h
            ) for f in filters
        ])
        
        # Branch 3: Objectness (Confidence)
        self.obj_convs = nn.ModuleList([
            nn.Sequential(
                Conv(f, f, 3, 1),
                Conv(f, f, 3, 1),
                nn.Conv2d(f, 1, 1)  # Confidence score
            ) for f in filters
        ])

    def forward(self, x):
        """
        Args:
            x: List of [P3, P4, P5] feature maps
        
        Returns:
            predictions: (batch, anchors, 85) for each scale
        """
        z = []  # Output
        for i, xi in enumerate(x):
            # P3 (80×80): ~42,400 predictions
            # P4 (40×40): ~10,600 predictions
            # P5 (20×20): ~2,600 predictions
            # Total: ~55,600 per image (before post-processing)
            
            cls = self.cls_convs[i](xi)  # (B, 80, H, W)
            bbox = self.reg_convs[i](xi)  # (B, 4, H, W)
            obj = self.obj_convs[i](xi)  # (B, 1, H, W)
            
            # Reshape to (B, H×W, 85)
            pred = torch.cat([bbox, obj, cls], 1)
            z.append(pred.permute(0, 2, 3, 1))
        
        return z
```

### Output Structure

#### Task 1: Classification Branch
```
Output: (B, H×W, 80)

For each grid cell in P3 (80×80 = 6,400 cells):
├─ Probability of "car": 0.95
├─ Probability of "pedestrian": 0.02
├─ Probability of "cyclist": 0.01
├─ ...
└─ Probability of "motorcycle": 0.005

For each grid cell in P4 (40×40 = 1,600 cells):
├─ (80 class probabilities)

For each grid cell in P5 (20×20 = 400 cells):
├─ (80 class probabilities)
```

**Example for car at (100, 150):**
```
Grid cell: (100/8, 150/8) = (12, 18) in P3
Predictions:
  - Car: 0.92 ✓ (matches ground truth)
  - Bus: 0.05
  - Pedestrian: 0.02
  - Others: < 0.01
```

#### Task 2: Regression (Bounding Box) Branch
```
Output: (B, H×W, 4)

For each grid cell:
├─ Δx: -0.3 (offset from grid center, normalized)
├─ Δy: 0.1
├─ Δw: 0.8 (log-scaled width)
└─ Δh: 0.6 (log-scaled height)

Decoding:
  Actual Box Center X = (grid_x + sigmoid(Δx)) × stride
  Actual Box Center Y = (grid_y + sigmoid(Δy)) × stride
  Actual Width = exp(Δw) × anchor_width
  Actual Height = exp(Δh) × anchor_height
```

**Example for car detection:**
```
Grid cell: (12, 18) at P3 (stride=8)
Center X = (12 + 0.7) × 8 = 101.6 pixels
Center Y = (18 + 0.6) × 8 = 148.8 pixels
Width = exp(0.8) × base_w ≈ 120 pixels
Height = exp(0.6) × base_h ≈ 60 pixels
→ Box: [41.6, 118.8, 161.6, 178.8]
```

#### Task 3: Objectness (Confidence) Branch
```
Output: (B, H×W, 1)

For each grid cell:
└─ Confidence score (0-1): 0.88

Interpretation:
  - 0.88 = 88% confidence that an object exists here
  - Combined with classification probability
  - Final score = objectness × class_probability
```

### "Decoupled" Design Explanation

**Coupled (Old YOLOv3/v4):**
```
[Shared conv tower]
    ↓
[Task 1: Classify]  [Task 2: Regress]  [Task 3: Objectness]
All competing for same features → Conflicts
```

**Decoupled (YOLOv11):**
```
[Shared backbone]
    ↓ P3, P4, P5
    ├─→ [Dedicated Classification Branch] (learns class features)
    ├─→ [Dedicated Regression Branch] (learns spatial features)
    └─→ [Dedicated Objectness Branch] (learns object presence)

No competition → Each task gets optimized features
```

### Output Dimensions for BDD100K

```
Input: 1280×720 image

P3 (80×80, 128 channels):
  Predictions: 80 × 80 × 85 = 544,000 values

P4 (40×40, 256 channels):
  Predictions: 40 × 40 × 85 = 136,000 values

P5 (20×20, 512 channels):
  Predictions: 20 × 20 × 85 = 34,000 values

Total Raw Predictions: 714,000 values ≈ 8,422 potential boxes

After NMS (Confidence > 0.45, IoU > 0.5):
  Typical: 100-300 final detections per image
```

---

## 5. Loss Functions

### 1. CIoU Loss (Bounding Box Regression)

**Purpose:** Ensure predicted boxes match ground truth exactly

$$\text{CIoU Loss} = 1 - \text{IoU} + \frac{\rho^2(b, \hat{b})}{c^2} + \alpha \cdot v$$

Where:
- **IoU (Intersection over Union):** Overlap ratio
- **ρ²(b, b̂):** Distance between centers squared
- **c²:** Diagonal of smallest enclosing box squared
- **α·v:** Aspect ratio penalty

**Example:**
```
Predicted Box: [50, 100, 150, 180]
Ground Truth:  [48, 102, 152, 178]

IoU = 0.91 (91% overlap)
Distance penalty: 0.02 (centers close)
Aspect ratio penalty: 0.01 (similar aspect ratios)

Total Loss: 1 - 0.91 + 0.02 + 0.01 = 0.12
```

**For BDD100K:** Penalizes not just size, but also position and aspect ratio consistency

### 2. Focal Loss (Classification)

**Purpose:** Handle class imbalance (many background → few foreground)

$$\text{Focal Loss} = -\alpha \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

Where:
- **p_t:** Model's predicted probability for true class
- **γ (gamma):** 2.0 (focus parameter, hardness factor)
- **α (alpha):** 0.25 (class weight)

**Effect:**
```
Easy examples (p_t = 0.9):  Loss = -0.25 × 0.1^2 × log(0.9) ≈ 0.002 (minimal)
Hard examples (p_t = 0.1):  Loss = -0.25 × 0.9^2 × log(0.1) ≈ 0.52 (emphasized)

Result: Hard negatives (mistaken backgrounds) get more focus
```

**BDD100K Benefit:** Focuses on difficult traffic scenarios (occlusions, small objects)

### 3. DFL Loss (Distribution Focal Loss)

**Purpose:** Better predict box edges using probability distributions

Instead of predicting single values (x, y, w, h), predict probability distributions:
```
Traditional: Predict width = 120 pixels
DFL: Predict probability distribution over possible widths
  - 118 pixels: 0.15
  - 119 pixels: 0.25
  - 120 pixels: 0.35 ← Most likely
  - 121 pixels: 0.20
  - 122 pixels: 0.05
```

**Advantage:** Smoother predictions, better calibration, easier confidence estimation

---

## 6. Complete Forward Pass Example

### Input
```python
# Image from BDD100K dataset
image = torch.randn(1, 3, 1280, 720)  # (batch=1, channels=3, H, W)
```

### Processing

```python
# === BACKBONE ===
# Stage 1-4: Multi-scale feature extraction
p1 = backbone.stem(image)  # (1, 32, 640, 360)
p2 = backbone.stage1(p1)   # (1, 64, 320, 180)
p3 = backbone.stage2(p2)   # (1, 128, 160, 90)
p4 = backbone.stage3(p3)   # (1, 256, 80, 45)
p5 = backbone.stage4(p4)   # (1, 512, 40, 22)
p5 = backbone.sppf(p5)     # (1, 512, 40, 22) - enhanced

# === NECK (Top-Down Fusion) ===
# Reduce channels and upsample from P5 to P4
p4_td = upsample(p5)       # (1, 256, 80, 44)
p4_td = concatenate([p4, p4_td])  # (1, 512, 80, 44)
p4_td = c2f_block(p4_td)   # (1, 256, 80, 44)

# Further reduce and upsample from P4 to P3
p3_td = upsample(p4_td)    # (1, 128, 160, 88)
p3_td = concatenate([p3, p3_td])  # (1, 256, 160, 88)
p3_td = c2f_block(p3_td)   # (1, 128, 160, 88)

# === NECK (Bottom-Up Fusion) ===
# Downsample and fuse from P3 to P4
p4_bu = downsample(p3_td)  # (1, 128, 80, 44)
p4_bu = concatenate([p4_td, p4_bu])  # (1, 384, 80, 44)
p4_bu = c2f_block(p4_bu)   # (1, 256, 80, 44)

# Further downsample and fuse from P4 to P5
p5_bu = downsample(p4_bu)  # (1, 256, 40, 22)
p5_bu = concatenate([p5, p5_bu])  # (1, 768, 40, 22)
p5_bu = c2f_block(p5_bu)   # (1, 512, 40, 22)

# === HEAD ===
# Task 1: Classification
cls_p3 = cls_head[0](p3_td)  # (1, 80, 160, 88) → (1, 160, 88, 80)
cls_p4 = cls_head[1](p4_bu)  # (1, 80, 80, 44) → (1, 80, 44, 80)
cls_p5 = cls_head[2](p5_bu)  # (1, 80, 40, 22) → (1, 40, 22, 80)

# Task 2: Regression (Bounding Box)
bbox_p3 = bbox_head[0](p3_td)  # (1, 4, 160, 88) → (1, 160, 88, 4)
bbox_p4 = bbox_head[1](p4_bu)  # (1, 4, 80, 44) → (1, 80, 44, 4)
bbox_p5 = bbox_head[2](p5_bu)  # (1, 4, 40, 22) → (1, 40, 22, 4)

# Task 3: Objectness
obj_p3 = obj_head[0](p3_td)  # (1, 1, 160, 88) → (1, 160, 88, 1)
obj_p4 = obj_head[1](p4_bu)  # (1, 1, 80, 44) → (1, 80, 44, 1)
obj_p5 = obj_head[2](p5_bu)  # (1, 1, 40, 22) → (1, 40, 22, 1)

# === POST-PROCESSING ===
# Combine predictions
predictions = []

for i in range(3):  # P3, P4, P5
    # Shape: (H×W, 85) where 85 = 4 bbox + 1 obj + 80 cls
    pred = concat([bbox[i], obj[i], cls[i]], dim=-1)
    predictions.append(pred)

# Total raw predictions
total_anchors = (160*88) + (80*44) + (40*22)  # ≈ 16,128
total_predictions = total_anchors * 85  # ≈ 1,370,880 values

# === NMS (Non-Maximum Suppression) ===
# Filter by objectness confidence
high_conf = predictions[predictions[:, 4] > 0.45]  # Keep confident

# Apply NMS: suppress overlapping boxes (IoU > 0.5)
final_boxes = nms(high_conf, iou_threshold=0.5)

# === OUTPUT ===
# Final detections: (num_detections, 6)
# Format: [x1, y1, x2, y2, confidence, class_id]
# Example output for 150 detections:
# [[100, 120, 280, 210, 0.92, 2],   # Car
#  [500, 340, 600, 420, 0.88, 0],   # Pedestrian
#  [150, 200, 250, 260, 0.81, 5],   # Cyclist
#  ...]
```

### Key Numbers for BDD100K

| Metric | Value |
|--------|-------|
| Input resolution | 1280×720 |
| Backbone parameters | ~20.9M |
| Total model parameters | ~25.5M |
| FLOPs (640×640) | ~86.7B |
| Inference time (RTX A6000) | ~8-12 ms |
| Raw predictions per image | ~16,128 boxes |
| After NMS | ~100-300 boxes |

---

## 7. BDD100K Specific Considerations

### Class Distribution
```
10 detection classes in BDD100K:
1. Car (60% of detections)
2. Pedestrian (15%)
3. Cyclist (8%)
4. Traffic Light (7%)
5. Traffic Sign (5%)
6. Truck (2%)
7. Bus (1%)
8. Train (0.5%)
9. Motorcycle (0.5%)
10. Other (1%)
```

**YOLOv11 handles this via:**
- Focal Loss to focus on rare classes (Train, Motorcycle)
- SPPF to capture train/truck sizes (10x size variation)
- Decoupled head to specialize classification separately

### Scale Variation
```
Object sizes in pixels at 1280×720:

Pedestrian: 40-120 pixels → P3 (80×80)
Cyclist: 50-150 pixels → P3/P4
Car: 60-300 pixels → P4 (40×40)
Truck: 100-400 pixels → P4/P5
Train: 200-600 pixels → P5 (20×20)
```

**PANet handles this via:**
- Top-down fusion: Large object context to small object features
- Bottom-up fusion: Small object details to large object features

### Occlusion & Truncation
```
Partially visible car: YOLOv11 still detects via
- Classification branch: Remains confident with visible parts
- Regression branch: DFL loss handles uncertain boundaries
- Objectness branch: Maintains confidence if enough features visible
```

---

## 8. Training Configuration for BDD100K

```python
model = YOLOv11m()  # Medium variant

# Loss weights
loss_weights = {
    'cls': 1.0,     # Classification loss
    'bbox': 7.5,    # Bounding box (CIoU) loss
    'obj': 1.0,     # Objectness loss
    'dfl': 1.5,     # Distribution focal loss
}

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005
)

# Learning rate schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.0001
)

# Augmentation
augmentation = {
    'hsv_h': 0.015,      # HSV hue
    'hsv_s': 0.7,        # HSV saturation
    'hsv_v': 0.4,        # HSV value
    'degrees': 10.0,     # Rotation
    'translate': 0.1,    # Translation
    'scale': 0.5,        # Scale
    'flipud': 0.0,       # Vertical flip
    'fliplr': 0.5,       # Horizontal flip (common in traffic)
    'mosaic': 1.0,       # Mosaic augmentation
    'mixup': 0.0,        # Mixup
}
```

---

## 9. Performance Metrics on BDD100K

### Achieved Results
```
Validation Set (100k images):
├─ mAP@0.5 (IoU > 50%): 0.5415 (54.15%)
├─ mAP@0.5:0.95 (IoU > 50% to 95%): 0.3131 (31.31%)
├─ Precision: 0.7190 (71.90%)
├─ Recall: 0.4966 (49.66%)
└─ Inference time: ~8-10 ms per image

Per-class performance:
├─ Car: 0.68 mAP@0.5 (largest class)
├─ Pedestrian: 0.45 mAP@0.5 (challenging)
├─ Cyclist: 0.42 mAP@0.5 (small, rare)
└─ Others: 0.25-0.40 mAP@0.5 (rare classes)
```

### Key Insights
1. **High Precision (71.9%):** Few false positives in high-confidence predictions
2. **Moderate Recall (49.7%):** Misses ~50% of objects (improvement opportunity)
3. **Recall-Precision Gap:** Loosening confidence threshold increases recall but decreases precision
4. **Class Imbalance:** Rare classes (Train, Motorcycle) have lower mAP

---

## 10. References

1. **YOLOv11 Paper:** "You Only Look Once: Unified, Real-Time Object Detection" (2024)
2. **CSPDarknet:** "CSPNet: A New Backbone that can Enhance Learning Capability of CNN"
3. **PANet:** "Path Aggregation Network for Instance Segmentation"
4. **Focal Loss:** "Focal Loss for Dense Object Detection" (ICCV 2017)
5. **DFL:** "Generalized Focal Loss: Learning to Locate Objects with Uncertainty"
6. **BDD100K Dataset:** "BDD100K: A Diverse Driving Video Database" (ICCV 2019)

---

## 11. Quick Reference: YOLOv11 Variants

```
YOLOv11n (Nano):       2.6M parameters,  6.3M FLOPs,  ~2 ms
YOLOv11s (Small):      9.6M parameters, 21.7M FLOPs,  ~4 ms
YOLOv11m (Medium):    20.9M parameters, 86.7M FLOPs,  ~8 ms  ← Used in this project
YOLOv11l (Large):     35.5M parameters,206.8M FLOPs, ~12 ms
YOLOv11x (XLarge):    56.9M parameters,477.2M FLOPs, ~15 ms
```

For BDD100K: Medium (11m) provides good balance between speed and accuracy.

---

## 12. Debugging Guide

### Common Issues

**Issue: Low Recall (missing objects)**
```
Likely causes:
1. Confidence threshold too high → Increase to 0.3
2. Model underfitting → Train longer, more data
3. NMS IoU threshold too strict → Increase to 0.6

Diagnosis:
- Check P3 branch: Are small objects detected at all?
- Check P4 branch: Are medium objects detected?
- If raw model good but NMS bad → Fix NMS threshold
```

**Issue: Low Precision (false positives)**
```
Likely causes:
1. Confidence threshold too low → Increase to 0.5
2. Class imbalance → Use Focal Loss weights
3. Overlapping objects confused → Improve training data

Diagnosis:
- Check which classes have false positives
- Analyze NMS overlap regions
- Consider class-specific confidence thresholds
```

**Issue: Slow inference**
```
Solution: Use smaller model variant
YOLOv11s: ~4x faster than 11m
Trade-off: ~5% accuracy loss for 4x speedup
```

