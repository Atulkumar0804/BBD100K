"""
Visualization of Model Predictions

Compare ground truth vs predictions with visualizations of:
- Correct detections
- False positives
- False negatives (missed detections)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import json
import os
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PredictionVisualizer:
    """Visualize model predictions vs ground truth."""
    
    CLASSES = [
        'person', 'rider', 'car', 'truck', 'bus',
        'train', 'motor', 'bike', 'traffic light', 'traffic sign'
    ]
    
    # Colors: BGR format for OpenCV
    GT_COLOR = (0, 255, 0)  # Green for ground truth
    PRED_COLOR = (0, 0, 255)  # Red for predictions
    TP_COLOR = (0, 255, 0)  # Green for true positives
    FP_COLOR = (0, 0, 255)  # Red for false positives
    FN_COLOR = (255, 0, 0)  # Blue for false negatives
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ):
        """
        Initialize visualizer.
        
        Args:
            model_path: Path to trained model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for matching
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_predictions(self, pred_boxes, pred_labels, gt_boxes, gt_labels):
        """
        Match predictions with ground truth boxes.
        
        Returns:
            tp_indices, fp_indices, fn_indices
        """
        tp_pred = []
        fp_pred = []
        fn_gt = list(range(len(gt_boxes)))
        matched_gt = set()
        
        for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if pred_label != gt_label:
                    continue
                
                if gt_idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                tp_pred.append(pred_idx)
                matched_gt.add(best_gt_idx)
                if best_gt_idx in fn_gt:
                    fn_gt.remove(best_gt_idx)
            else:
                fp_pred.append(pred_idx)
        
        return tp_pred, fp_pred, fn_gt
    
    def visualize_comparison(
        self,
        image_path: str,
        ground_truth: Dict,
        output_path: str = None
    ):
        """
        Visualize ground truth vs predictions side by side.
        
        Args:
            image_path: Path to image
            ground_truth: Dict with 'boxes' and 'labels'
            output_path: Where to save visualization
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            return
        
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        # Extract predictions
        pred_boxes = []
        pred_labels = []
        pred_confs = []
        
        for box in results.boxes:
            pred_boxes.append(box.xyxy[0].cpu().numpy())
            pred_labels.append(int(box.cls[0]))
            pred_confs.append(float(box.conf[0]))
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Ground Truth
        ax = axes[0]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title('Ground Truth', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        for box, label in zip(ground_truth['boxes'], ground_truth['labels']):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, self.CLASSES[label],
                   bbox=dict(facecolor='green', alpha=0.7),
                   fontsize=10, color='white')
        
        # Right: Predictions
        ax = axes[1]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title('Predictions', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        for box, label, conf in zip(pred_boxes, pred_labels, pred_confs):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{self.CLASSES[label]} {conf:.2f}",
                   bbox=dict(facecolor='red', alpha=0.7),
                   fontsize=10, color='white')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        
        plt.close()
    
    def visualize_errors(
        self,
        image_path: str,
        ground_truth: Dict,
        output_path: str = None
    ):
        """
        Visualize true positives, false positives, and false negatives.
        
        Args:
            image_path: Path to image
            ground_truth: Dict with 'boxes' and 'labels'
            output_path: Where to save visualization
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            return
        
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        # Extract predictions
        pred_boxes = []
        pred_labels = []
        pred_confs = []
        
        for box in results.boxes:
            pred_boxes.append(box.xyxy[0].cpu().numpy())
            pred_labels.append(int(box.cls[0]))
            pred_confs.append(float(box.conf[0]))
        
        # Match predictions
        tp_indices, fp_indices, fn_indices = self.match_predictions(
            pred_boxes, pred_labels,
            ground_truth['boxes'], ground_truth['labels']
        )
        
        # Draw on image
        vis_image = image.copy()
        
        # Draw True Positives (Green)
        for idx in tp_indices:
            box = pred_boxes[idx]
            label = pred_labels[idx]
            conf = pred_confs[idx]
            x1, y1, x2, y2 = [int(v) for v in box]
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), self.TP_COLOR, 2)
            cv2.putText(vis_image, f"TP: {self.CLASSES[label]} {conf:.2f}",
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TP_COLOR, 2)
        
        # Draw False Positives (Red)
        for idx in fp_indices:
            box = pred_boxes[idx]
            label = pred_labels[idx]
            conf = pred_confs[idx]
            x1, y1, x2, y2 = [int(v) for v in box]
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), self.FP_COLOR, 2)
            cv2.putText(vis_image, f"FP: {self.CLASSES[label]} {conf:.2f}",
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.FP_COLOR, 2)
        
        # Draw False Negatives (Blue)
        for idx in fn_indices:
            box = ground_truth['boxes'][idx]
            label = ground_truth['labels'][idx]
            x1, y1, x2, y2 = [int(v) for v in box]
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), self.FN_COLOR, 2)
            cv2.putText(vis_image, f"FN: {self.CLASSES[label]}",
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.FN_COLOR, 2)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "TP (True Positive)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TP_COLOR, 2)
        cv2.putText(vis_image, "FP (False Positive)", (10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.FP_COLOR, 2)
        cv2.putText(vis_image, "FN (False Negative)", (10, legend_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.FN_COLOR, 2)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            print(f"Saved error visualization to {output_path}")
        
        return vis_image, len(tp_indices), len(fp_indices), len(fn_indices)


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize predictions vs ground truth')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--gt-json', type=str, required=True, help='Path to ground truth JSON')
    parser.add_argument('--output', type=str, default='output-Data_Analysis/visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    # Load ground truth
    with open(args.gt_json, 'r') as f:
        gt_data = json.load(f)
    
    # Find ground truth for this image
    image_name = os.path.basename(args.image)
    ground_truth = None
    
    for item in gt_data:
        if item.get('name') == image_name:
            # Extract boxes and labels
            boxes = []
            labels = []
            for label_info in item.get('labels', []):
                if label_info.get('category') in PredictionVisualizer.CLASSES:
                    box2d = label_info.get('box2d')
                    if box2d:
                        boxes.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
                        labels.append(PredictionVisualizer.CLASSES.index(label_info['category']))
            
            ground_truth = {'boxes': boxes, 'labels': labels}
            break
    
    if ground_truth is None:
        print(f"Ground truth not found for {image_name}")
        return
    
    # Initialize visualizer
    visualizer = PredictionVisualizer(args.model)
    
    # Create visualizations
    output_comparison = os.path.join(args.output, f"{image_name}_comparison.png")
    visualizer.visualize_comparison(args.image, ground_truth, output_comparison)
    
    output_errors = os.path.join(args.output, f"{image_name}_errors.png")
    visualizer.visualize_errors(args.image, ground_truth, output_errors)
    
    print("✅ Visualizations complete!")


if __name__ == "__main__":
    main()
