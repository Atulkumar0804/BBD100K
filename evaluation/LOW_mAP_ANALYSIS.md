# Low mAP@0.5 Analysis Report

## Issue Summary

Expected: **mAP@0.5 = 0.54** (54%)  
Actual: **mAP@0.5 = 0.166** (16.6%)  
**Difference: -37.4 percentage points**

---

## Root Cause Analysis

### 1. Class-wise Performance Breakdown

| Class | AP | Precision | Recall | GT Objects | Predictions | Status |
|-------|-----|-----------|--------|------------|-------------|--------|
| **person** | 0.52 | 0.76 | 0.52 | 746 | 513 | ✅ Good |
| **truck** | 0.51 | 0.62 | 0.54 | 212 | 185 | ✅ Good |
| **car** | 0.09 | 0.04 | 0.0002 | 5,062 | 24 | ❌ SEVERE |
| **bus** | 0.09 | 0.002 | 0.09 | 91 | 4,436 | ❌ SEVERE |
| **bike** | 0.09 | 0.0 | 0.0 | 40 | 71 | ❌ No detections |
| **motor** | 0.0 | 0.0 | 0.0 | 20 | 0 | ❌ No detections |
| **traffic light** | 0.09 | 0.0 | 0.0 | 1,374 | 16 | ❌ No detections |
| **traffic sign** | 0.09 | 0.0 | 0.0 | 1,754 | 28 | ❌ No detections |
| **train** | 0.09 | 0.0 | 0.0 | 1 | 590 | ❌ No detections |
| **rider** | 0.09 | 0.0 | 0.0 | 35 | 1,306 | ❌ No detections |

### 2. Key Problems

**A. Catastrophic Underprediction (Missing Objects):**
```
car:          24 predictions vs 5,062 ground truth    (99.5% miss rate)
traffic light: 16 predictions vs 1,374 ground truth   (98.8% miss rate)
traffic sign:  28 predictions vs 1,754 ground truth   (98.4% miss rate)
```

**B. Severe Overprediction (False Positives):**
```
bus:   4,436 predictions vs 91 ground truth     (48.7x false positive rate)
rider: 1,306 predictions vs 35 ground truth     (37.3x false positive rate)
train: 590 predictions vs 1 ground truth        (590x false positive rate!)
```

**C. Complete Failure on Small Objects:**
- Traffic lights, signs, and bikes: 0% recall
- Model cannot detect small objects at all

---

## Likely Causes

### 1. **Model Mismatch**
The `runs-model/best.pt` file might be:
- Not trained on BDD100K dataset
- Trained on different image size (e.g., 640 but BDD100K is 1280×720)
- From a different YOLO version
- Not properly trained (stopped early)

### 2. **Dataset vs Model Incompatibility**
```
BDD100K Characteristics:
- Image size: 1280×720
- Small objects: 30-40% of annotations (traffic lights, signs)
- Class distribution: Imbalanced (many cars, few trains)

Model Prediction Behavior:
- Excellent on large objects (person, truck)
- Fails on small objects (traffic light, sign)
- Confused class predictions (600 train detections for 1 ground truth)
```

### 3. **Inference Configuration Issues**
- Confidence threshold may be wrong
- NMS (Non-Maximum Suppression) parameter incompatible
- Anchor boxes not matching dataset

### 4. **Training Data Mismatch**
The expected 0.54 mAP was likely achieved on:
- Different validation set
- Different training setup
- Different input resolution

---

## Evidence from Metrics

### Recall Analysis (% of ground truth objects detected)

```
person:        52%  ✅ Reasonable
truck:         54%  ✅ Reasonable
car:           0.02% ❌ Almost nothing detected
traffic light:  0%  ❌ Nothing detected
bike:           0%  ❌ Nothing detected
```

### Precision Analysis (% of predictions correct)

```
person:        76%  ✅ Mostly correct predictions
truck:         62%  ✅ Mostly correct predictions
car:            4%  ❌ 96 out of 100 predictions are wrong
bus:           0.2% ❌ 99.8% of predictions are wrong
train:          0%  ❌ All 590 predictions are garbage
```

---

## Solutions to Investigate

### 1. **Verify Model Training**
```bash
# Check if model was trained on BDD100K
# Look for training logs in runs-model/
ls -lh runs-model/*/training_summary.json

# Check training results
cat runs-model/*/training_summary.json | grep -i "final_metrics\|mAP"
```

### 2. **Retrain Model (Recommended)**
```bash
cd model/
python3 train.py --epochs 50
# This will create runs-model/bdd100k_yolo11_50epochs_*

# Then evaluate the new model
cd ../evaluation/
python3 run_model_eval.py --yolo_weights ../runs-model/bdd100k_yolo11_*/weights/best.pt --max_images 2000
```

### 3. **Test with Ultralytics Pretrained Model**
```bash
# Download standard YOLOv11 pretrained on COCO
# Then fine-tune on BDD100K

python3 << 'EOF'
from ultralytics import YOLO

# Load pretrained YOLOv11 (trained on COCO)
model = YOLO("yolov11m.pt")

# Fine-tune on BDD100K
results = model.train(
    data="configs/bdd100k.yaml",
    epochs=50,
    imgsz=640,  # or 1280
    device=0
)
EOF
```

### 4. **Debug Inference Output**
```bash
# Check what the model is actually predicting
python3 << 'EOF'
from ultralytics import YOLO
import numpy as np
from PIL import Image

model = YOLO("runs-model/best.pt")

# Test on a single image
image = Image.open("data/bdd100k/images/100k/val/image.jpg")
results = model.predict(image, conf=0.25, verbose=True)

# Print predictions
for result in results:
    print(f"Found {len(result.boxes)} objects")
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        print(f"  Class: {cls} (conf: {conf:.2f}) Box: {box}")
EOF
```

---

## Comparison with Expected Metrics

| Metric | Expected | Actual | Gap |
|--------|----------|--------|-----|
| mAP@0.5 | 0.541 | 0.166 | -37.5% |
| Precision (avg) | ~0.72 | 0.19* | -53% |
| Recall (avg) | ~0.50 | 0.14* | -36% |
| Per-class consistency | Balanced | Highly imbalanced | ❌ |

*Weighted by class frequency

---

## Recommendations (Priority Order)

### 🔴 Critical
1. **Verify the trained model actually exists and is on BDD100K**
   - Check: `runs-model/bdd100k_yolo11_*/training_summary.json`
   - Look for: `"mAP@0.5": 0.54` in training logs

### 🟠 High Priority
2. **Retrain the model properly**
   ```bash
   cd model/
   python3 train.py --epochs 50 --batch_size 16
   ```

3. **Use correct model weights**
   - Ensure you're using the best trained model, not a pretrained one

### 🟡 Medium Priority
4. **Optimize inference configuration**
   - Try different confidence thresholds: 0.1, 0.25, 0.5
   - Adjust NMS threshold if needed

### 🟢 Low Priority  
5. **Data preprocessing**
   - Verify images are in correct format
   - Check if images need resizing/normalization

---

## Command to Generate Detailed Report

```bash
# Evaluate with detailed reporting
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

# Low confidence (catches more objects, more false positives)
python3 evaluation/run_model_eval.py --max_images 1000 --conf_threshold 0.1

# High confidence (fewer objects, more precision)
python3 evaluation/run_model_eval.py --max_images 1000 --conf_threshold 0.5

# Compare results
cat evaluation/metrics/summary.json
cat evaluation/metrics/yolov11/metrics.json | head -50
```

---

## Next Steps

1. **Immediately**: Verify if model was trained on BDD100K
2. **Today**: Retrain model with 50+ epochs
3. **Then**: Re-run evaluation and compare metrics
4. **Finally**: Debug individual classes with lowest performance

If the retrained model also has low mAP, the issue may be:
- Dataset preprocessing (labels format, image format)
- Model architecture not suitable for small objects
- Data augmentation needs tuning

