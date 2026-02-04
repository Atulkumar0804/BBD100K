# BDD100K Data Analysis

This folder contains the data analysis pipeline for the BDD100K **object detection** labels. It parses the BDD100K JSON labels, computes summary statistics, exports analysis results as JSON, generates plots, and optionally serves a Streamlit dashboard.

## What this pipeline does

1. **Parse labels** into clean dataclasses for images and bounding boxes.
2. **Compute analysis metrics** (class balance, bbox sizes, anomalies, etc.).
3. **Save results** to `output-Data_Analysis/analysis_results.json`.
4. **Generate plots** under `output-Data_Analysis/visualizations/` and sample images under `output-Data_Analysis/interesting_samples/`.
5. **View results** in a Streamlit dashboard with tabs and image galleries.

## Folder inputs & outputs (expected)

**Inputs** (typically in `../data`):
- `data/labels/train.json`
- `data/labels/val.json`
- `data/images/train/`
- `data/images/val/`

**Outputs** (typically in `../output-Data_Analysis`):
- `output-Data_Analysis/analysis_results.json`
- `output-Data_Analysis/visualizations/*.png`
- `output-Data_Analysis/interesting_samples/*.jpg`

> Paths are configurable via CLI arguments; see commands below.

## Key insights (from analysis)

- Traffic lights are predominantly tiny objects.
- Cars dominate instance count → class imbalance.
- Val split slightly underrepresents rare classes.
- Crowded scenes concentrate many objects in a few images.

These findings motivated the choice of a multi-scale detector in the modeling stage.

## What’s new in the latest pipeline
- **Exact image counts**: label totals vs disk totals.
- **Missing-label detection**: images on disk without label entries.
- **Empty-label detection**: label entries with no valid boxes.
- **High-DPI plots**: visualizations saved at higher resolution.
- **Expanded plots**: pie charts, CDFs, co-occurrence, spatial heatmaps.
- **Dashboard galleries**: 2-column image grids with per-image explanations.

## How the scripts connect

```
parser.py  ──>  analysis.py  ──>  visualize.py  ──>  dashboard.py
                  │                 │
                  └── results JSON ─┘
```

## File-by-file responsibilities

### `parser.py`
- Defines dataclasses: `BoundingBox`, `ImageAnnotation`.
- Defines `DETECTION_CLASSES` list (10 detection classes).
- `parse_bdd_json()` reads BDD100K label JSON and returns structured annotations.
- `iter_all_objects()` yields all boxes across images.

### `analysis.py`
- Core analysis pipeline (`BDD100KAnalyzer`).
- Computes:
  - **Class distribution** (instances and images per class).
  - **Objects per image** distribution.
  - **BBox sizes** (stats + size buckets).
  - **Train/val balance** ratios.
  - **Anomalies** (empty images, tiny boxes, crowded scenes).
  - **Interesting samples** (largest/smallest per class, crowded, rare-only).
- Saves results to JSON for downstream plotting and dashboard.
- Adds image-level counts (labels vs disk), missing label entries, and empty label entries.

### `visualize.py`
- Generates static plots from `analysis_results.json` (high DPI).
- Exports example images with drawn boxes for “interesting samples”.
- Produces files under:
  - `output-Data_Analysis/visualizations/`
  - `output-Data_Analysis/interesting_samples/`

### `dashboard.py`
- Streamlit app with tabs for metrics, tables, plots, and samples.
- Loads `output-Data_Analysis/analysis_results.json` and renders a 2-column image gallery.
- Shows a short explanation under each plot/sample.

### `convert_to_yolo.py`
- Utility to convert BDD100K JSON labels to YOLO-format `.txt` labels.
- Outputs per-image label files in YOLO format (normalized coordinates).
- This is **preprocessing**, not analysis, but is kept here for dataset prep.

### `Dockerfile`
- Container definition for running the analysis pipeline in isolation.

## Typical workflow

### 1) Run analysis

```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection
python3 data_analysis/analysis.py --dataset_dir data --output_dir output-Data_Analysis
```

### 2) Generate plots and sample images

```bash
python3 data_analysis/visualize.py --dataset_dir data --output_dir output-Data_Analysis
```

### 3) Launch dashboard

```bash
streamlit run data_analysis/dashboard.py
```

## CLI options (quick reference)

### `analysis.py`
- `--dataset_dir`: root dataset directory (default `data`)
- `--output_dir`: output directory (default `output-Data_Analysis`)

### `visualize.py`
- `--dataset_dir`: root dataset directory (default `data`)
- `--output_dir`: output directory (default `output-Data_Analysis`)

### `convert_to_yolo.py`
- `--train-json`: path to train label JSON
- `--val-json`: path to val label JSON
- `--train-output`: output dir for train YOLO labels
- `--val-output`: output dir for val YOLO labels

## Dependencies

The analysis stack uses common scientific and visualization libraries (e.g., `numpy`, `pandas`, `matplotlib`, `seaborn`, `streamlit`, `Pillow`). Install from project requirements, typically:

```bash
pip install -r requirements_analysis.txt
```

---


