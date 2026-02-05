# Model Documentation

## 1. Model Choice: YOLOv11

**Why YOLOv11?**
Based on the analysis of the BDD100K dataset, we selected YOLOv11 for the following reasons:
*   **Real-world Applicability:** YOLO (You Only Look Once) is the industry standard for autonomous driving due to its inference speed vs. accuracy balance.
*   **Small Object Detection:** The dataset contains many small objects (traffic lights, distant signs). YOLOv11's improved feature pyramid network (FPN) handles multi-scale detection better than previous iterations.
*   **Multi-Object Handling:** BDD100K images often contain dozens of cars and pedestrians. YOLO is designed to predict dense bounding boxes efficiently.

## 2. Architecture Overview

The YOLOv11 architecture consists of three main components:

### 1️⃣ Backbone (Feature Extraction)
*   **Structure:** CSPDarknet / EfficientNet-based custom backbone.
*   **Function:** It acts as a Convolutional Neural Network (CNN) that processes the input image to extract features. It learns low-level features (edges, textures) in early layers and high-level semantic features (shapes, objects) in deeper layers.

### 2️⃣ Neck (Feature Aggregation)
*   **Structure:** Path Aggregation Network (PANet) / FPN.
*   **Function:** This is the critical part for BDD100K. It combines feature maps from different scales (high resolution for small objects, low resolution for large objects). This allows the model to detect a traffic light (small) and a bus (large) in the same image with high accuracy.

### 3️⃣ Head (Prediction)
*   **Structure:** Decoupled Head.
*   **Function:** The head makes the final predictions.
    *   **Regression Branch:** Predicts the bounding box coordinates (cx, cy, w, h).
    *   **Classification Branch:** Predicts the probability of each class (Car, Pedestrian, etc.).

### Loss Functions
*   **Box Loss:** CIoU (Complete Intersection over Union) - penalizes aspect ratio inconsistencies.
*   **Class Loss:** Binary Cross Entropy (BCE) for classification.
*   **DFL (Distribution Focal Loss):** Improves localization accuracy.

## 3. Training Strategy

*   **Framework:** Ultralytics YOLO.
*   **Pre-trained Weights:** We initialized the model with `yolo11n.pt` (COCO pre-trained) to leverage transfer learning, reducing convergence time.
*   **Epochs:** Trained for 10 epochs.
*   **Image Size:** Resized to 640x640 (standard YOLO input).
*   **Data Augmentation:** Mosaic augmentation was used to force the model to learn context and handle occlusions (common in traffic scenes).

## 4. Limitations

*   **Compute Constraints:** Training was limited to 10 epochs. A full production run would typically require 50-300 epochs for optimal convergence.
*   **Dataset Subset:** While we used the 100k dataset structure, hyperparameter tuning was done on a limited scope.
