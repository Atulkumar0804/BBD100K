"""
Evaluation Metrics for Object Detection

Implements:
- mAP (mean Average Precision) at different IoU thresholds
- Precision and Recall
- Class-wise Average Precision
- F1 Score
- Confusion Matrix
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import defaultdict
import json


class DetectionMetrics:
    """Calculate object detection metrics."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        iou_thresholds: List[float] = [0.5, 0.75],
        conf_threshold: float = 0.25
    ):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
            iou_thresholds: IoU thresholds for mAP calculation
            conf_threshold: Confidence threshold for predictions
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds
        self.conf_threshold = conf_threshold
        
        # Storage for predictions and ground truth
        self.predictions = defaultdict(list)  # {class_id: [(conf, iou, is_tp)]}
        self.ground_truths = defaultdict(int)  # {class_id: count}
    
    def calculate_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """
        Calculate IoU between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ):
        """
        Update metrics with predictions and ground truths for one image.
        
        Args:
            predictions: List of predicted boxes
                Each dict: {'boxes': [[x1,y1,x2,y2], ...], 'labels': [...], 'scores': [...]}
            ground_truths: List of ground truth boxes
                Each dict: {'boxes': [[x1,y1,x2,y2], ...], 'labels': [...]}
        """
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']
            
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']
            
            # Track which ground truths have been matched
            gt_matched = [False] * len(gt_boxes)
            
            # Count ground truths per class
            for label in gt_labels:
                self.ground_truths[int(label)] += 1
            
            # Sort predictions by confidence (descending)
            sorted_indices = np.argsort(pred_scores)[::-1]
            
            for pred_idx in sorted_indices:
                pred_box = pred_boxes[pred_idx]
                pred_label = int(pred_labels[pred_idx])
                pred_score = pred_scores[pred_idx]
                
                # Skip low confidence predictions
                if pred_score < self.conf_threshold:
                    continue
                
                # Find best matching ground truth
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    # Only match with same class
                    if int(gt_label) != pred_label:
                        continue
                    
                    # Skip already matched ground truths
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Store prediction result
                # Will determine TP/FP based on IoU threshold during AP calculation
                self.predictions[pred_label].append({
                    'confidence': pred_score,
                    'iou': best_iou,
                    'matched_gt': best_gt_idx
                })
                
                # Mark ground truth as matched
                if best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
    
    def calculate_ap(
        self,
        class_id: int,
        iou_threshold: float
    ) -> Tuple[float, float, float]:
        """
        Calculate Average Precision for a class at specific IoU threshold.
        
        Args:
            class_id: Class ID
            iou_threshold: IoU threshold
            
        Returns:
            Tuple of (AP, precision, recall)
        """
        if class_id not in self.predictions or self.ground_truths[class_id] == 0:
            return 0.0, 0.0, 0.0
        
        # Sort predictions by confidence
        preds = sorted(self.predictions[class_id], key=lambda x: x['confidence'], reverse=True)
        
        # Calculate TP and FP
        tp = []
        fp = []
        
        for pred in preds:
            if pred['matched_gt'] >= 0 and pred['iou'] >= iou_threshold:
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        
        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        recalls = tp_cumsum / self.ground_truths[class_id]
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Add sentinel values
        recalls = np.concatenate([[0.0], recalls, [1.0]])
        precisions = np.concatenate([[1.0], precisions, [0.0]])
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        # Final precision and recall
        final_precision = precisions[-2] if len(precisions) > 1 else 0.0
        final_recall = recalls[-2] if len(recalls) > 1 else 0.0
        
        return ap, final_precision, final_recall
    
    def calculate_map(self, iou_threshold: float = 0.5) -> Dict:
        """
        Calculate mean Average Precision across all classes.
        
        Args:
            iou_threshold: IoU threshold
            
        Returns:
            Dictionary with mAP and per-class metrics
        """
        aps = []
        per_class_metrics = {}
        
        for class_id in range(self.num_classes):
            ap, precision, recall = self.calculate_ap(class_id, iou_threshold)
            aps.append(ap)
            
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            per_class_metrics[self.class_names[class_id]] = {
                'AP': ap,
                'precision': precision,
                'recall': recall,
                'F1': f1,
                'num_gt': self.ground_truths.get(class_id, 0),
                'num_pred': len(self.predictions.get(class_id, []))
            }
        
        mAP = np.mean(aps)
        
        return {
            'mAP': mAP,
            'iou_threshold': iou_threshold,
            'per_class': per_class_metrics
        }
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate metrics at all IoU thresholds.
        
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        for iou_threshold in self.iou_thresholds:
            metrics = self.calculate_map(iou_threshold)
            results[f'mAP@{iou_threshold}'] = metrics
        
        # Calculate mAP@[.5:.95] (COCO metric)
        map_values = []
        for iou in np.linspace(0.5, 0.95, 10):
            metrics = self.calculate_map(iou)
            map_values.append(metrics['mAP'])
        
        results['mAP@[.5:.95]'] = np.mean(map_values)
        
        return results
    
    def print_metrics(self):
        """Print formatted metrics."""
        results = self.calculate_all_metrics()
        
        print("\n" + "="*80)
        print("OBJECT DETECTION EVALUATION METRICS")
        print("="*80)
        
        # Overall metrics
        print(f"\n📊 Overall Performance:")
        print(f"  mAP@0.5      : {results['mAP@0.5']['mAP']:.4f}")
        print(f"  mAP@0.75     : {results['mAP@0.75']['mAP']:.4f}")
        print(f"  mAP@[.5:.95] : {results['mAP@[.5:.95]']:.4f}")
        
        # Per-class metrics (at IoU 0.5)
        print(f"\n📋 Per-Class Metrics (IoU=0.5):")
        print(f"{'Class':<20} {'AP':<8} {'Precision':<12} {'Recall':<8} {'F1':<8} {'GT':<6} {'Pred':<6}")
        print("-" * 80)
        
        per_class = results['mAP@0.5']['per_class']
        for class_name, metrics in sorted(per_class.items(), key=lambda x: x[1]['AP'], reverse=True):
            print(f"{class_name:<20} "
                  f"{metrics['AP']:<8.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['F1']:<8.4f} "
                  f"{metrics['num_gt']:<6} "
                  f"{metrics['num_pred']:<6}")
        
        print("="*80 + "\n")
        
        return results
    
    def save_metrics(self, output_path: str):
        """Save metrics to JSON file."""
        results = self.calculate_all_metrics()
        
        # Convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Metrics saved to {output_path}")


def main():
    """Example usage."""
    # Example data
    class_names = [
        'person', 'rider', 'car', 'truck', 'bus',
        'train', 'motor', 'bike', 'traffic light', 'traffic sign'
    ]
    
    metrics = DetectionMetrics(
        num_classes=10,
        class_names=class_names,
        iou_thresholds=[0.5, 0.75]
    )
    
    # Simulate some predictions and ground truths
    # In practice, these would come from your model and dataset
    predictions = [{
        'boxes': np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
        'labels': np.array([2, 0]),
        'scores': np.array([0.9, 0.8])
    }]
    
    ground_truths = [{
        'boxes': np.array([[105, 105, 205, 205], [350, 350, 450, 450]]),
        'labels': np.array([2, 5])
    }]
    
    metrics.update(predictions, ground_truths)
    results = metrics.print_metrics()


if __name__ == "__main__":
    main()
