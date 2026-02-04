"""
Error Analysis Module

Analyzes model failures and clusters them by:
- Object size (small, medium, large)
- Lighting conditions (day, night, dawn/dusk)
- Occlusion level
- Class-specific patterns
"""

import numpy as np
import json
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


class ErrorAnalyzer:
    """Analyze and categorize detection errors."""
    
    CLASSES = [
        'person', 'rider', 'car', 'truck', 'bus',
        'train', 'motor', 'bike', 'traffic light', 'traffic sign'
    ]
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize error analyzer.
        
        Args:
            model_path: Path to trained model
            conf_threshold: Confidence threshold
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Error tracking
        self.errors = {
            'by_size': {'small': [], 'medium': [], 'large': []},
            'by_lighting': {'day': [], 'night': [], 'dawn_dusk': []},
            'by_occlusion': {'none': [], 'partial': [], 'heavy': []},
            'by_class': {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in self.CLASSES}
        }
    
    def categorize_size(self, bbox: List[float]) -> str:
        """Categorize object by size (COCO definition)."""
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        if area < 32 * 32:
            return 'small'
        elif area < 96 * 96:
            return 'medium'
        else:
            return 'large'
    
    def detect_lighting(self, image_path: str) -> str:
        """
        Detect lighting condition from image.
        
        Simple heuristic based on average brightness.
        """
        image = cv2.imread(image_path)
        if image is None:
            return 'unknown'
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 60:
            return 'night'
        elif avg_brightness < 120:
            return 'dawn_dusk'
        else:
            return 'day'
    
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
    
    def analyze_image(
        self,
        image_path: str,
        ground_truth: Dict,
        iou_threshold: float = 0.5
    ):
        """
        Analyze errors for a single image.
        
        Args:
            image_path: Path to image
            ground_truth: Dict with 'boxes', 'labels', 'attributes'
            iou_threshold: IoU threshold for matching
        """
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        # Extract predictions
        pred_boxes = []
        pred_labels = []
        
        for box in results.boxes:
            pred_boxes.append(box.xyxy[0].cpu().numpy())
            pred_labels.append(int(box.cls[0]))
        
        # Detect lighting
        lighting = self.detect_lighting(image_path)
        
        # Track matched ground truths
        matched_gt = set()
        
        # Analyze predictions (TP and FP)
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(
                zip(ground_truth['boxes'], ground_truth['labels'])
            ):
                if pred_label != gt_label:
                    continue
                
                if gt_idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True Positive
                self.errors['by_class'][self.CLASSES[pred_label]]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False Positive
                self.errors['by_class'][self.CLASSES[pred_label]]['fp'] += 1
                
                # Categorize FP
                size_cat = self.categorize_size(pred_box)
                self.errors['by_size'][size_cat].append({
                    'type': 'fp',
                    'class': self.CLASSES[pred_label],
                    'lighting': lighting
                })
        
        # Analyze False Negatives (missed ground truths)
        for gt_idx, (gt_box, gt_label) in enumerate(
            zip(ground_truth['boxes'], ground_truth['labels'])
        ):
            if gt_idx not in matched_gt:
                # False Negative
                self.errors['by_class'][self.CLASSES[gt_label]]['fn'] += 1
                
                # Categorize FN
                size_cat = self.categorize_size(gt_box)
                self.errors['by_size'][size_cat].append({
                    'type': 'fn',
                    'class': self.CLASSES[gt_label],
                    'lighting': lighting
                })
                
                # Check occlusion
                if 'attributes' in ground_truth:
                    attrs = ground_truth.get('attributes', {})
                    occluded = attrs.get('occluded', False)
                    occlusion_level = 'partial' if occluded else 'none'
                else:
                    occlusion_level = 'none'
                
                self.errors['by_occlusion'][occlusion_level].append({
                    'type': 'fn',
                    'class': self.CLASSES[gt_label],
                    'lighting': lighting
                })
    
    def generate_report(self, output_dir: str = 'output-Data_Analysis/error_analysis'):
        """Generate comprehensive error analysis report."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("ERROR ANALYSIS REPORT")
        print("="*80)
        
        # Class-wise analysis
        print("\n📊 Class-wise Performance:")
        print(f"{'Class':<20} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<8}")
        print("-" * 80)
        
        for cls, metrics in self.errors['by_class'].items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            print(f"{cls:<20} {tp:<8} {fp:<8} {fn:<8} {precision:<12.4f} {recall:<8.4f}")
        
        # Size-based analysis
        print("\n📏 Errors by Object Size:")
        for size, errors in self.errors['by_size'].items():
            fp_count = sum(1 for e in errors if e['type'] == 'fp')
            fn_count = sum(1 for e in errors if e['type'] == 'fn')
            print(f"  {size.capitalize()}: FP={fp_count}, FN={fn_count}")
        
        # Lighting analysis
        print("\n💡 Errors by Lighting:")
        lighting_errors = defaultdict(lambda: {'fp': 0, 'fn': 0})
        
        for size_cat in self.errors['by_size'].values():
            for error in size_cat:
                lighting = error.get('lighting', 'unknown')
                error_type = error['type']
                lighting_errors[lighting][error_type] += 1
        
        for lighting, counts in lighting_errors.items():
            print(f"  {lighting.capitalize()}: FP={counts['fp']}, FN={counts['fn']}")
        
        # Occlusion analysis
        print("\n🔍 Errors by Occlusion:")
        for occlusion, errors in self.errors['by_occlusion'].items():
            fn_count = len([e for e in errors if e['type'] == 'fn'])
            print(f"  {occlusion.capitalize()}: FN={fn_count}")
        
        print("\n" + "="*80)
        
        # Generate visualizations
        self.plot_error_distribution(output_dir)
        
        # Save detailed report
        self.save_detailed_report(output_dir)
    
    def plot_error_distribution(self, output_dir: str):
        """Create visualization plots for error distribution."""
        # Size distribution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Errors by size
        ax = axes[0, 0]
        sizes = ['small', 'medium', 'large']
        fp_counts = [sum(1 for e in self.errors['by_size'][s] if e['type'] == 'fp') for s in sizes]
        fn_counts = [sum(1 for e in self.errors['by_size'][s] if e['type'] == 'fn') for s in sizes]
        
        x = np.arange(len(sizes))
        width = 0.35
        
        ax.bar(x - width/2, fp_counts, width, label='False Positives', color='red', alpha=0.7)
        ax.bar(x + width/2, fn_counts, width, label='False Negatives', color='blue', alpha=0.7)
        ax.set_xlabel('Object Size')
        ax.set_ylabel('Count')
        ax.set_title('Errors by Object Size')
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in sizes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Class-wise precision-recall
        ax = axes[0, 1]
        precisions = []
        recalls = []
        class_names = []
        
        for cls, metrics in self.errors['by_class'].items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            if tp + fp + fn > 0:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                class_names.append(cls)
        
        ax.scatter(recalls, precisions, s=100, alpha=0.6)
        for i, cls in enumerate(class_names):
            ax.annotate(cls, (recalls[i], precisions[i]), fontsize=8)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Class-wise Precision vs Recall')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Plot 3: TP/FP/FN per class
        ax = axes[1, 0]
        classes = list(self.errors['by_class'].keys())
        tp_counts = [self.errors['by_class'][c]['tp'] for c in classes]
        fp_counts = [self.errors['by_class'][c]['fp'] for c in classes]
        fn_counts = [self.errors['by_class'][c]['fn'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, tp_counts, width, label='TP', color='green', alpha=0.7)
        ax.bar(x, fp_counts, width, label='FP', color='red', alpha=0.7)
        ax.bar(x + width, fn_counts, width, label='FN', color='blue', alpha=0.7)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Detection Results per Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: F1 scores
        ax = axes[1, 1]
        f1_scores = []
        
        for cls in classes:
            metrics = self.errors['by_class'][cls]
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        ax.barh(classes, f1_scores, color='steelblue', alpha=0.7)
        ax.set_xlabel('F1 Score')
        ax.set_title('F1 Score per Class')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        output_path = f"{output_dir}/error_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Error distribution plot saved to {output_path}")
        plt.close()
    
    def save_detailed_report(self, output_dir: str):
        """Save detailed JSON report."""
        report = {
            'by_class': self.errors['by_class'],
            'by_size': {
                size: {
                    'fp': sum(1 for e in errors if e['type'] == 'fp'),
                    'fn': sum(1 for e in errors if e['type'] == 'fn')
                }
                for size, errors in self.errors['by_size'].items()
            },
            'by_occlusion': {
                occ: len([e for e in errors if e['type'] == 'fn'])
                for occ, errors in self.errors['by_occlusion'].items()
            }
        }
        
        output_path = f"{output_dir}/error_report.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Detailed report saved to {output_path}")


def main():
    """Example usage."""
    print("Error Analysis Module")
    print("This module analyzes detection errors by size, lighting, and occlusion.")
    print("\nUsage:")
    print("  analyzer = ErrorAnalyzer('path/to/model.pt')")
    print("  analyzer.analyze_image(image_path, ground_truth)")
    print("  analyzer.generate_report()")


if __name__ == "__main__":
    main()
