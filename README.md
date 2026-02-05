# BDD100K Object Detection - Bosch Applied CV Assignment

**Author:** Atul  
**Date:** February 2026  
**Task:** End-to-end object detection pipeline on BDD100K dataset

## Project Overview

This project implements a complete computer vision pipeline for object detection on the Berkeley DeepDrive 100K (BDD100K) dataset. The solution is designed to be modular, reproducible, and scalable, covering the entire lifecycle from data exploration to model deployment and visualization.

Key features include:
- Comprehensive data analysis and statistical reporting.
- YOLOv11-based object detection model implementation.
- Automated training pipeline with logging and checkpointing.
- Extensive evaluation metrics (mAP, Precision, Recall, Confusion Matrices).
- Interactive dashboard for visualizing dataset statistics and model performance.
- Containerized environment support for reproducibility.

## Project Structure

The project is organized into modular components for clarity and maintainability:

```text
bosch-bdd-object-detection/
|-- configs/                   # Configuration files (e.g., YOLO dataset YAML)
|-- data/                      # Dataset directory (input images and labels)
|-- data_analysis/             # Scripts for data exploration and reporting
|   |-- analysis.py            # Generates statistical analysis of the dataset
|   |-- convert_to_yolo.py     # Converts BDD100K JSON labels to YOLO format
|   |-- dashboard.py           # Streamlit dashboard application
|   |-- parser.py              # Utilities for parsing dataset annotations
|   |-- visualize.py           # Generates static visualization plots
|
|-- evaluation/                # Scripts for model evaluation
|   |-- metrics.py             # core metric calculations
|   |-- error_analysis.py      # Detailed error analysis tools
|   |-- run_model_eval.py      # Main evaluation entry point
|   |-- visualize_predictions.py # Generates prediction overlays
|   |-- MODEL_PERFORMANCE_REPORT.md # Generated report
|
|-- model/                     # Model training and inference code
|   |-- model.py               # Model architecture definition
|   |-- train.py               # Main training script
|   |-- inference.py           # Inference script for testing
|   |-- dataset_loader.py      # Custom dataset loaders
|
|-- notebooks/                 # Jupyter notebooks for interactive exploration
|-- output-Data_Analysis/      # Generated artifacts (plots, JSON reports)
|-- runs-model/                # Model training outputs (weights, logs)
|-- requirements.txt           # Python dependency requirements
|-- docker-compose.yml         # Container orchestration configuration
|-- README.md                  # Project documentation
```

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM recommended for data processing

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd bosch-bdd-object-detection
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

Place the BDD100K dataset in the `data/` directory matching the following structure:
```text
data/bdd100k/
|-- images/
|   |-- 100k/
|       |-- train/
|       |-- val/
|-- labels/
    |-- bdd100k_labels_images_train.json
    |-- bdd100k_labels_images_val.json
```

## Usage Guide

### 1. Data Analysis
Run the full analysis pipeline to generate statistics and visualizations.
```bash
python data_analysis/analysis.py
python data_analysis/visualize.py
```
This generates `analysis_results.json` and plots in `output-Data_Analysis/`.

### 2. Interactive Dashboard
Launch the Streamlit dashboard to explore the dataset and model results interactively.
```bash
streamlit run data_analysis/dashboard.py
```
The dashboard provides tabs for:
- Dataset Overview & Metrics
- Data Tables (Class balance, BBox stats)
- Analysis Visualizations
- Scene & Object Attributes
- Model Evaluation Performance
- Sample Images with Metadata Overlays

### 3. Training the Model
To train the YOLOv11 model:
```bash
python model/train.py
```
Training artifacts (weights, logs, curves) are saved to `runs-model/`.

### 4. Evaluation
Run the evaluation suite to calculate metrics and generate a performance report.
```bash
python evaluation/run_model_eval.py
```

## Results

Final analysis and model artifacts are stored in the following directories:
- **Metrics & Analysis**: `output-Data_Analysis/`
- **Trained Weights**: `runs-model/bdd100k_yolo11_*/weights/`
- **Performance Plots**: `runs-model/bdd100k_yolo11_*/`

For detailed performance metrics, refer to `evaluation/MODEL_PERFORMANCE_REPORT.md` after running the evaluation script.

## Docker Support

The project includes a comprehensive Docker setup to ensure reproducibility across different environments. A single unified container is used to manage all services: data analysis, training, and visualization.

### Building the Image
Build the Docker image using the provided Compose configuration:
```bash
docker-compose build
```

### Running Services
You can run individual components of the pipeline using Docker Compose:

**Run Data Analysis:**
```bash
docker-compose up analysis
```

**Train YOLOv11 Model (Requires GPU):**
```bash
docker-compose up train-yolo
```

**Launch Dashboard:**
```bash
docker-compose up dashboard
```
Access the dashboard at `http://localhost:8501`.

**Launch Jupyter Notebooks:**
```bash
docker-compose up jupyter
```
Access Jupyter at `http://localhost:8888`.

## License
This project is part of the Bosch Applied CV Assignment.
