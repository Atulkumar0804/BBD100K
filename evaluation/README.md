# Evaluation & Analysis Report

This directory collects tools and reports for evaluating the performance of the object detection models.

## 1. Metrics Evaluation

We evaluated the YOLOv11 model using standard object detection metrics.

### Quantitative Results (Validation Set)
Based on the training run (checking `best.pt` performance at epoch 20):

*   **mAP@0.5:** **0.541** (54.1%) - This indicates that for detections with at least 50% overlap with ground truth, the model provides decent average precision.
*   **Precision:** **0.719** (71.9%) - When the model predicts an object, it is correct 71.9% of the time. This suggests a low rate of False Positives.
*   **Recall:** **0.496** (49.6%) - The model correctly identifies 49.6% of all ground truth objects. This suggests a higher rate of False Negatives (missed objects), likely due to small objects or occlusions.

### Training vs. Validation Loss
Processing the `results.csv` logs:
*   **Box Loss:** Decreased consistently (Train: 1.37 -> 1.19, Val: 1.34 -> 1.18). The model is learning to localize objects better over time.
*   **Classification Loss:** Decreased (Train: 0.93 -> 0.63, Val: 0.88 -> 0.62). The model is getting better at distinguishing between cars, people, and lights.

## 2. Confusion Matrix Analysis (Why is "Background" included?)

You may notice a **Background** row/column in the confusion matrix. This is critical for object detection analysis:

*   **"Background" is NOT a class** in the labeled dataset (like 'car' or 'person').
*   **Ground Truth = Class X, Predicted = Background:** This represents a **False Negative (Missed Detection)**. The object was there, but the model "saw" background.
*   **Ground Truth = Background, Predicted = Class X:** This represents a **False Positive (Ghost Detection)**. The model predicted an object where there was empty space.

**Evaluation of our matrix:**
*   High values in the `(Class X, Background)` column indicate the model is missing many objects (Low Recall).
*   High values in the `(Background, Class X)` row indicate the model is hallucinating objects (Low Precision).

## 3. Qualitative Evaluation (Visual Inspection)

We ran inference on the **Test Dataset**. Results are saved in `output-Data_Analysis/test_predictions`.

### Observations:
*   **Successes:** The model robustly detects **Cars** and **Trucks**, especially in daylight. Large objects are rarely missed.
*   **Failures (Clusters):**
    1.  **Small Objects:** Distant **Traffic Lights** and **Traffic Signs** are frequently missed (contributing to low Recall).
    2.  **Occlusion:** Pedestrians partially hidden behind cars are sometimes classified as background.
    3.  **Night Scenes:** Performance drops in low-light conditions due to lower contrast.

## 4. Suggested Improvements

To improve the Recall (currently 48%) and overall mAP:

1.  **Data Augmentation:** Increase identifying small objects by using *Copy-Paste* augmentation or *MixUp* to oversample smaller classes like traffic signs.
2.  **Training Duration:** Train for 50+ epochs. The loss curves were still decreasing, indicating the model had not yet converged.
3.  **Input Resolution:** Increase input size from 640x640 to 1280x1280 to give the model more pixels to resolve small traffic lights.

## 5. Scripts Description

*   **`metrics.py`**: Computes precision, recall, and mAP metrics.
*   **`visualize_predictions.py`**: Runs inference on images and overlays prediction boxes for visual inspection.
*   **`error_analysis.py`**: Analyzes prediction errors, clustering them by object size, occlusion, and lighting conditions.
*   **`run_model_eval.py`**: Orchestration script to run the full evaluation suite.
