# Data Analysis Pipeline - Detailed Documentation

This directory contains a **complete data analysis and visualization pipeline** for the BDD100K object detection dataset. It provides comprehensive tools for parsing annotations, generating statistical reports, creating publication-quality visualizations, and hosting an interactive web-based dashboard.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Pipeline Overview](#pipeline-overview)
3. [Running the Pipeline](#running-the-pipeline)
4. [File-by-File Detailed Explanation](#file-by-file-detailed-explanation)
5. [Dashboard Features](#dashboard-features)
6. [Output Artifacts](#output-artifacts)
7. [Data Flow Diagram](#data-flow-diagram)

---

## Quick Start

### Prerequisites

```bash
# Ensure Python 3.10+ is installed
python3 --version

# Install required packages
pip install numpy pandas matplotlib seaborn pillow streamlit plotly ultralytics
```

### Run Complete Pipeline (5 minutes)

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

# Step 1: Parse and analyze dataset
python3 data_analysis/analysis.py

# Step 2: Generate visualizations
python3 data_analysis/visualize.py

# Step 3: Launch interactive dashboard
streamlit run data_analysis/dashboard.py
```

**Expected Output:**
- `output-Data_Analysis/analysis_results.json` - Statistical report
- `output-Data_Analysis/visualizations/` - 20+ PNG plots
- `output-Data_Analysis/interesting_samples/` - 100+ annotated images
- Dashboard at `http://localhost:8501`

### Individual Commands

#### 1. Parse Dataset and Generate Statistics
```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

python3 data_analysis/analysis.py \
    --train_json data/bdd100k/labels/bdd100k_labels_images_train.json \
    --val_json data/bdd100k/labels/bdd100k_labels_images_val.json \
    --output_dir output-Data_Analysis

# Output: analysis_results.json (5-10 MB)
```

#### 2. Generate Visualizations
```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

python3 data_analysis/visualize.py \
    --analysis_json output-Data_Analysis/analysis_results.json \
    --images_dir data/bdd100k/images/100k \
    --output_dir output-Data_Analysis/visualizations

# Output: 20+ PNG plots + 100+ annotated images
```

#### 3. Launch Interactive Dashboard
```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

streamlit run data_analysis/dashboard.py

# Open: http://localhost:8501 (Ctrl+C to stop)
```

#### 4. Convert to YOLO Format (for model training)
```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

python3 data_analysis/convert_to_yolo.py \
    --train_json data/bdd100k/labels/bdd100k_labels_images_train.json \
    --val_json data/bdd100k/labels/bdd100k_labels_images_val.json \
    --train_output data/bdd100k/labels/100k/train \
    --val_output data/bdd100k/labels/100k/val

# Output: .txt files ready for YOLOv11 training
```

#### 5. Download Dataset (if missing)
```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

python3 data_analysis/download_dataset.py

# Downloads and extracts BDD100K to data/bdd100k/
```

---

## Pipeline Overview

The pipeline performs the following **5 structured stages**:

1. **📥 Parsing** → Reads raw BDD100K JSON files and converts them into typed Python dataclasses for type safety and clarity
2. **Analysis** → Computes comprehensive statistics: class distribution, bounding box sizes, object counts, anomalies, metadata attributes
3. **Visualization** → Generates 20+ high-quality publication-ready plots and annotated sample images
4. **🌐 Dashboard** → Interactive Streamlit web app for real-time exploration of dataset and model results
5. **Format Conversion** → Converts BDD100K JSON labels to YOLO `.txt` format for model training

### Key Statistics Computed
- **Class Distribution**: Instance counts + image counts per class (train/val)
- **Bounding Box Analysis**: Area distributions, size categories (small/medium/large), aspect ratios
- **Object Density**: Objects-per-image histograms and statistics
- **Dataset Metadata**: Weather, time-of-day, scene-type distributions
- **Object Attributes**: Occlusion and truncation rates per class
- **Imbalance Metrics**: Train/Val ratios, recommended class weights
- **Anomaly Detection**: Empty labels, tiny objects, unusual aspect ratios

---

## Running the Pipeline

### Complete Workflow (Recommended)

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

# Run all steps in sequence
echo "Step 1: Analyzing dataset..."
python3 data_analysis/analysis.py

echo "Step 2: Generating visualizations..."
python3 data_analysis/visualize.py

echo "Step 3: Starting dashboard (Ctrl+C to stop)..."
streamlit run data_analysis/dashboard.py
```

### Performance Expectations

| Stage | Time | RAM | Output Size |
|-------|------|-----|------------|
| Parsing (70k images) | 5-10 sec | 500 MB | N/A |
| Analysis | 15-30 sec | 1 GB | 5-10 MB |
| Visualizations | 1-2 min | 2 GB | 500 MB |
| Dashboard load | Instant | 300 MB | N/A |

### Expected Dataset Statistics

- **Total Images**: 70,000 training + 10,000 validation
- **Total Objects**: 450,000+ annotations
- **Classes**: 10 (person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign)
- **Image Resolution**: 1280×720 pixels
- **Empty Labels**: < 1% (data quality good)
- **Tiny Objects**: ~2-3% (small but trainable)

---

## File-by-File Detailed Explanation

### 1. `parser.py` — JSON Parser & Data Structures
**Purpose**: Parse BDD100K's JSON format into clean, type-safe Python dataclasses

#### Key Classes:
```python
@dataclass
class BoundingBox:
    """Single 2D bounding box for an object."""
    class_name: str           # One of 10 classes (person, car, bus, etc.)
    x1: int                   # Top-left X coordinate (pixels)
    y1: int                   # Top-left Y coordinate (pixels)
    x2: int                   # Bottom-right X coordinate (pixels)
    y2: int                   # Bottom-right Y coordinate (pixels)
    area: int                 # Pre-computed area (width × height)
    occluded: bool            # Is object occluded by another?
    truncated: bool           # Is object cut off by image boundary?

@dataclass
class ImageAnnotation:
    """Annotations for one image in the dataset."""
    image_name: str           # Filename (e.g., "0000f77c-6257be58.jpg")
    width: int                # Image width (typically 1280)
    height: int               # Image height (typically 720)
    objects: List[BoundingBox]  # List of all objects in this image
    weather: str              # Weather condition (clear, rainy, snowy, etc.)
    scene: str                # Scene type (highway, city street, parking lot, etc.)
    timeofday: str            # Time of day (day, night, dawn, dusk)
```

#### Functions:

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `parse_bdd_json()` | Path to JSON file | `List[ImageAnnotation]` | Parse entire JSON into typed objects |
| `iter_all_objects()` | Annotations list | Generator of `BoundingBox` | Iterate over all objects across all images |
| `_safe_int()` | Float/int/None | int | Convert numeric values safely with fallback |
| `_extract_resolution()` | JSON dict | (width, height) tuple | Extract image dimensions, default to 1280×720 |

#### Example Usage:
```python
from parser import parse_bdd_json, iter_all_objects

# Parse entire training set
annotations = parse_bdd_json("data/bdd100k/labels/bdd100k_labels_images_train.json")

# Iterate over all objects
for obj in iter_all_objects(annotations):
    print(f"{obj.class_name}: area={obj.area}, occluded={obj.occluded}")

# Access specific image
img = annotations[0]
print(f"Image: {img.image_name}, Weather: {img.weather}, Objects: {len(img.objects)}")
```

#### Special Features:
- **Type Safety**: Dataclasses prevent runtime errors and aid IDE autocomplete
- **Defensive Parsing**: Handles malformed JSON gracefully (missing fields default to safe values)
- **Class Filtering**: Can restrict to only certain classes during parsing
- **Metadata Extraction**: Captures weather, scene, time-of-day attributes automatically

---

### 2. `analysis.py` — Statistical Analysis Engine
**Purpose**: Compute comprehensive dataset statistics and generate JSON report

#### Main Class: `BDD100KAnalyzer`

This class orchestrates the entire analysis pipeline:

```python
class BDD100KAnalyzer:
    def __init__(self, train_labels: Path, val_labels: Path)
    def load()                              # Load and parse JSON files
    def analyze_class_distribution()        # Class counts, imbalance
    def analyze_objects_per_image()         # Object density analysis
    def analyze_bbox_sizes()                # Bounding box statistics
    def analyze_split_balance()             # Train/val ratios
    def analyze_attributes()                # Weather/scene/time distributions
    def detect_anomalies()                  # Find problematic annotations
    def save_results(path)                  # Export JSON report
```

#### Key Analysis Functions:

| Function | Computes | Output |
|----------|----------|--------|
| `analyze_class_distribution()` | Instance counts per class, image counts per class | Train/Val imbalance, class-specific stats |
| `analyze_objects_per_image()` | Distribution of object counts | Mean, median, percentiles, histograms |
| `analyze_bbox_sizes()` | Bounding box area statistics | Per-class size distributions, buckets (small/medium/large) |
| `analyze_split_balance()` | Train vs Val ratios | Imbalance ratios, recommended class weights |
| `analyze_attributes()` | Weather, scene, time-of-day | Distribution counts for each attribute |
| `detect_anomalies()` | Unusual patterns | Empty labels, tiny objects, extreme aspect ratios |

#### Sample Output (JSON):
```json
{
  "class_distribution": {
    "train": {
      "instances": {"car": 50000, "person": 30000, ...},
      "images": {"car": 40000, "person": 25000, ...}
    },
    "val": {
      "instances": {"car": 12500, "person": 7500, ...}
    },
    "class_stats": {
      "car": {
        "train_avg_bbox_area": 45000,
        "train_images_with_class": 40000,
        "train_avg_objects_per_image": 1.25
      }
    }
  },
  "bbox_sizes": {
    "per_class": {
      "car": {
        "train": {"mean": 45000, "median": 42000, "std": 15000},
        "val": {"mean": 44500, "median": 41500}
      }
    },
    "size_buckets": {
      "train": {"small": 8000, "medium": 25000, "large": 17000},
      "val": {"small": 2000, "medium": 6000, "large": 4500}
    }
  },
  "anomalies": {
    "empty_labels_train": 2,
    "tiny_objects_train": 145,
    "extreme_aspect_ratios": 12
  }
}
```

#### Usage:
```python
from analysis import BDD100KAnalyzer

# Create analyzer
analyzer = BDD100KAnalyzer(
    train_labels=Path("data/bdd100k/labels/bdd100k_labels_images_train.json"),
    val_labels=Path("data/bdd100k/labels/bdd100k_labels_images_val.json")
)

# Run all analyses
analyzer.load()
analyzer.analyze_class_distribution()
analyzer.analyze_bbox_sizes()
analyzer.analyze_attributes()
analyzer.detect_anomalies()

# Save results
analyzer.save_results("output-Data_Analysis/analysis_results.json")
```

#### Anomaly Detection:
The `detect_anomalies()` function identifies:
- **Empty labels**: Images with no objects
- **Tiny objects**: Area < 32×32 pixels (hard to see and learn)
- **Extreme aspect ratios**: Width/height > 10 or < 0.1 (unusual shapes)
- **Truncated/occluded**: High rates of truncation or occlusion per class

---

### 3. `visualize.py` — Plot Generation Engine
**Purpose**: Generate 20+ publication-quality static plots and annotated sample images

#### Main Functions:

| Function | Generates | Output Format |
|----------|-----------|----------------|
| `plot_class_distribution()` | Grouped bar chart (train vs val) | `class_distribution.png` |
| `plot_class_distribution_pie()` | Pie charts with percentages | `class_distribution_pie.png` |
| `plot_ratio()` | Val/Train ratio per class | `val_train_ratio.png` |
| `plot_objects_per_image()` | Histogram of object counts | `objects_per_image.png` |
| `plot_bbox_size_distribution()` | Boxplot of bbox areas | `bbox_size_distribution.png` |
| `plot_bbox_area_histogram()` | Histogram with mean/median lines | `bbox_area_histogram.png` |
| `plot_bbox_size_buckets()` | Pie chart of size categories | `bbox_size_buckets_pie.png` |
| `plot_avg_bbox_area_per_class()` | Bar chart (train vs val) | `avg_bbox_area_per_class.png` |
| `plot_tiny_bbox_per_class()` | Count of small objects | `tiny_bbox_per_class.png` |
| `plot_aspect_ratio_scatter()` | Width vs height scatter | `aspect_ratio_scatter.png` |
| `plot_aspect_ratio_distribution()` | Distribution of aspect ratios | `aspect_ratio_distribution.png` |
| `plot_class_cooccurrence()` | Heatmap of co-occurring classes | `class_cooccurrence_matrix.png` |
| `plot_attribute_distributions()` | Weather/scene/time pie charts | `attribute_distribution.png` |
| `export_interesting_samples()` | Annotated sample images | `interesting_samples/` folder |

#### Interesting Samples Exported:
```
interesting_samples/
├── largest_car_123.jpg          # Image with largest car bbox
├── largest_person_456.jpg       # Image with largest person bbox
├── smallest_traffic_light_789.jpg
├── most_crowded.jpg             # Image with most objects
├── rare_class_only.jpg          # Image with only rare classes
└── ...
```

Each sample image has:
- Ground-truth bounding boxes drawn (color-coded by class)
- Metadata overlay (weather, time of day, scene type)
- Class labels on each box

#### Color Scheme (Consistent):
```
person:       Red (#FF6B6B)
rider:        Teal (#4ECDC4)
car:          Blue (#45B7D1)
truck:        Light Salmon (#FFA07A)
bus:          Mint (#98D8C8)
train:        Purple (#6C5CE7)
motor:        Yellow (#FDCB6E)
bike:         Light Blue (#74B9FF)
traffic light: Aqua (#55EFC4)
traffic sign:  Pink (#FD79A8)
```

#### Usage:
```python
from visualize import plot_class_distribution, export_interesting_samples
import json

# Load analysis results
with open("output-Data_Analysis/analysis_results.json") as f:
    results = json.load(f)

# Generate plots
plot_class_distribution(results, output_dir=Path("output-Data_Analysis/visualizations"))
plot_bbox_area_histogram(results, output_dir=Path("output-Data_Analysis/visualizations"))
plot_class_cooccurrence(results, output_dir=Path("output-Data_Analysis/visualizations"))

# Export interesting samples
export_interesting_samples(
    train_annotations=annotations,
    images_dir=Path("data/bdd100k/images/100k/train"),
    output_dir=Path("output-Data_Analysis/interesting_samples")
)
```

---

### 4. `convert_to_yolo.py` — Format Converter for YOLOv11
**Purpose**: Convert BDD100K JSON annotations to YOLO `.txt` format for model training

#### Format Conversion Details:

**BDD100K Format:**
```json
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
```

**YOLO Format (one file per image):**
```
# File: 0000f77c-6257be58.txt
2 0.15625 0.35417 0.15625 0.27778  # car: center_x, center_y, width, height (normalized)
8 0.75000 0.25000 0.10000 0.08333  # traffic_light
```

#### Conversion Logic:
```
1. BDD100K box: x1, y1 (top-left), x2, y2 (bottom-right) in pixels
2. Calculate center: x_center = (x1 + x2) / 2, y_center = (y1 + y2) / 2
3. Calculate size: width = x2 - x1, height = y2 - y1
4. Normalize: x_center /= img_width, y_center /= img_height, etc.
5. Clamp to [0, 1] to handle edge cases
6. Output: class_id x_center y_center width height (space-separated)
```

#### Class ID Mapping:
```python
CLASS_MAPPING = {
    'person': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motor': 6,
    'bike': 7,
    'traffic light': 8,
    'traffic sign': 9
}
```

#### Usage:
```bash
# Convert both train and val sets
python convert_to_yolo.py \
    --train-json data/bdd100k/labels/bdd100k_labels_images_train.json \
    --val-json data/bdd100k/labels/bdd100k_labels_images_val.json \
    --train-output data/bdd100k/labels/100k/train \
    --val-output data/bdd100k/labels/100k/val
```

Output:
```
Converted 70,000 training images → 70,000 .txt files
Converted 20,000 val images → 20,000 .txt files
Total objects: 450,000+ annotations
```

---

### 5. `dashboard.py` — Interactive Streamlit Web App
**Purpose**: Real-time web-based exploration of dataset and model performance

#### Architecture:
- **Non-compute intensive**: Reads pre-computed JSON results from `analysis.py`
- **6 Interactive Tabs**: Organized discovery experience
- **Auto-detection**: Finds latest model training run automatically
- **Gallery view**: Beautiful image galleries with consistent styling

#### Dashboard Tabs:

#### **Tab 1: Overview & Metrics**
Displays executive summary:
- Total image counts (train/val)
- Disk vs label discrepancies
- Dataset health indicators:
  - Empty labels count
  - Tiny objects (< 1024 px²)
  - Extreme aspect ratios
  - Truncation/occlusion rates

#### **Tab 2: Data Tables**
Interactive sortable tables:
- **Class Distribution Table**: 
  - Train instances, Val instances, Train images, Val images
  - Val/Train ratio (identifies imbalance)
  - Recommendations for class weights
- **Bounding Box Statistics**:
  - Mean/median area per class (train vs val)
  - Min/max sizes
- **Attribute Table**:
  - Occlusion % per class
  - Truncation % per class

#### **Tab 3: Analysis Plots**
Displays all pre-generated visualizations:
- Class distribution bar charts
- Bounding box size distributions
- Objects-per-image histograms
- Aspect ratio plots
- Attribute pie charts
- Co-occurrence heatmap

#### **Tab 4: Attributes**
BDD100K-specific metadata analysis:
- **Weather**: Distribution pie chart (clear, rainy, snowy, foggy)
- **Time of Day**: Distribution pie chart (day, night, dawn, dusk)
- **Scene Type**: Distribution pie chart (highway, city street, parking lot, etc.)
- **Object Attributes**:
  - Occlusion rates per class
  - Truncation rates per class

#### **Tab 5: Model Evaluation**
Automatically detects latest YOLOv11 training run:
- Training metrics (if available):
  - Loss curves (box, class, DFL)
  - mAP@0.5, mAP@0.75 over epochs
  - Precision/Recall curves
- Performance summary:
  - Best checkpoint metrics
  - Class-wise AP scores

#### **Tab 6: Sample Images**
Gallery of interesting edge cases:
- Most crowded images
- Largest/smallest objects per class
- Rare class examples
- Different lighting/weather conditions

Each image shows:
- Ground-truth boxes (color-coded)
- Class labels
- Metadata overlay (weather, time, scene)

#### Launch Dashboard:
```bash
cd bosch-bdd-object-detection
streamlit run data_analysis/dashboard.py
```

Then open: **http://localhost:8501**

#### Dashboard Features:
- **Responsive Design**: Works on desktop, tablet, mobile
- **Fast Loading**: Pre-computed results (no heavy computation)
- **Auto-refresh**: Detects new analysis results automatically
- **Exportable**: Download plots and tables as PNG/CSV
- **Session State**: Maintains user selections across tabs

---

### 6. `download_dataset.py` — Dataset Downloader
**Purpose**: Auto-download BDD100K dataset from cloud storage if not present locally

#### Features:
- **Smart Detection**: Checks if dataset already exists, skips if present
- **Resume Support**: Can resume incomplete downloads
- **Progress Bar**: Shows download progress with ETA
- **Auto-extract**: Extracts ZIP/TAR archives automatically
- **Validation**: Checks file integrity after download

#### Usage:
```bash
python data_analysis/download_dataset.py
```

#### Workflow:
```
1. Check if data/bdd100k/ exists
2. If missing:
   a. Download from Google Drive (using FILE_ID)
   b. Show progress bar with ETA
   c. Extract to data/bdd100k/
   d. Verify structure
3. Print summary
```

---

## Dashboard Features

### Interactive Filtering & Sorting
- Click column headers to sort tables
- Use dropdown filters (class, weather, etc.)
- Search across tables

### Visualization Interactions
- **Hover**: Show exact values on plots
- **Zoom**: Click and drag to zoom into regions
- **Download**: Click camera icon to save plots as PNG
- **Export**: Download tables as CSV

### Real-time Updates
Dashboard automatically detects:
- New analysis results (reloads JSON)
- Latest model training runs (shows newest metrics)
- New sample images (updates gallery)

---

## Output Artifacts

### Generated Files & Directories

```
output-Data_Analysis/
├── analysis_results.json          # Complete JSON report (statistics)
├── visualizations/                # 20+ publication-quality PNG plots
│   ├── class_distribution.png
│   ├── bbox_size_distribution.png
│   ├── objects_per_image.png
│   ├── aspect_ratio_scatter.png
│   ├── class_cooccurrence_matrix.png
│   ├── attribute_distribution.png
│   └── ... (15+ more plots)
└── interesting_samples/           # 100+ annotated edge-case images
    ├── largest_car_123.jpg
    ├── smallest_person_456.jpg
    ├── most_crowded.jpg
    ├── rare_class_only.jpg
    └── ... (many more samples)
```

### analysis_results.json Structure
```json
{
  "class_distribution": {...},      # Instance/image counts per class
  "bbox_sizes": {...},              # Size statistics and distributions
  "objects_per_image": {...},       # Density analysis
  "attribute_distribution": {...},  # Weather/scene/time counts
  "object_attributes": {...},       # Occlusion/truncation per class
  "split_balance": {...},           # Train/val ratios and imbalance
  "anomalies": {...}                # Empty labels, tiny objects, etc.
}
```

---

## Data Flow Diagram

```
Raw BDD100K Dataset
    ├─ images/100k/{train,val}/*.jpg
    └─ labels/bdd100k_labels_images_{train,val}.json
            ↓
    ┌──────────────────────────────────┐
    │ parser.py: Parse JSON            │
    │ → ImageAnnotation dataclasses    │
    │ → Type safety + validation       │
    └──────────────────────────────────┘
            ↓
    ┌──────────────────────────────────┐
    │ analysis.py: Compute Statistics  │
    │ → Class distributions            │
    │ → Bbox sizes & anomalies         │
    │ → Attribute analysis             │
    │ → Save JSON report               │
    └──────────────────────────────────┘
            ↓ (JSON Report)
    ┌──────────────────────────────────┐
    │ visualize.py: Generate Plots     │
    │ → 20+ static plots (PNG)         │
    │ → Annotated samples              │
    │ → Gallery exports                │
    └──────────────────────────────────┘
            ↓ (Plots + Samples)
    ┌──────────────────────────────────┐
    │ dashboard.py: Interactive UI     │
    │ → 6 exploration tabs             │
    │ → Real-time filtering            │
    │ → Model evaluation               │
    │ → http://localhost:8501          │
    └──────────────────────────────────┘

    Parallel Flow:
    ┌──────────────────────────────────┐
    │ convert_to_yolo.py: Format Conv  │
    │ → BDD100K JSON → YOLO .txt       │
    │ → Ready for model training       │
    └──────────────────────────────────┘
```

---

## Dependencies

The pipeline requires these Python libraries:

```
numpy          # Numerical computations
pandas         # Data manipulation & tables
matplotlib     # Plotting backend
seaborn        # Statistical plots
Pillow (PIL)   # Image processing
streamlit      # Interactive dashboard
plotly         # Interactive visualizations
ultralytics    # YOLO model integration
```

All included in `requirements.txt`.

---

## Performance Notes

| Stage | Typical Time | RAM Usage | Output Size |
|-------|-------------|-----------|------------|
| Parsing (70k images) | 5-10 sec | 500 MB | N/A |
| Analysis | 15-30 sec | 1 GB | 5-10 MB |
| Visualizations | 1-2 min | 2 GB | 500 MB |
| Dashboard load | Instant | 300 MB | N/A |
| YOLO conversion | 20-30 sec | 500 MB | 50 MB |

---

## Troubleshooting

### Issue: "JSON file not found"
**Solution**: Ensure `data/bdd100k/labels/` contains JSON files. Run `download_dataset.py` if missing.

### Issue: Dashboard takes long to load
**Solution**: This is normal first-time (pre-computing). Subsequent loads are instant.

### Issue: Visualizations look blurry
**Solution**: Plots are generated at 450 DPI for publication quality. Open with image viewer, not web browser.

### Issue: Out of memory during analysis
**Solution**: Process in batches or reduce dataset size. Modify `analysis.py` to limit image count.

---








