"""
Inference Module for YOLOv11 on BDD100K

Performs inference on images or videos and visualizes predictions.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import List, Union
import os


class BDD100KInference:
    """Inference class for trained YOLOv11 model."""
    
    CLASSES = [
        'person', 'rider', 'car', 'truck', 'bus',
        'train', 'motor', 'bike', 'traffic light', 'traffic sign'
    ]
    
    # Color map for visualization
    COLORS = {
        'person': (255, 0, 0),
        'rider': (255, 128, 0),
        'car': (0, 255, 0),
        'truck': (0, 255, 255),
        'bus': (0, 128, 255),
        'train': (0, 0, 255),
        'motor': (255, 0, 255),
        'bike': (255, 255, 0),
        'traffic light': (128, 0, 255),
        'traffic sign': (255, 192, 203)
    }
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize inference model.
        
        Args:
            model_path: Path to trained model weights
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"✅ Model loaded on {device}")
    
    def predict(
        self,
        source: Union[str, np.ndarray],
        save: bool = False,
        output_dir: str = 'output-Data_Analysis/predictions',
        show_labels: bool = True,
        show_conf: bool = True
    ):
        """
        Run prediction on image(s) or video.
        
        Args:
            source: Image path, folder path, or video path
            save: Whether to save predictions
            output_dir: Directory to save results
            show_labels: Show class labels
            show_conf: Show confidence scores
        """
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save,
            project=output_dir,
            show_labels=show_labels,
            show_conf=show_conf,
            verbose=True
        )
        
        return results
    
    def predict_and_visualize(
        self,
        image_path: str,
        output_path: str = None
    ):
        """
        Predict and visualize results on a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (if None, displays only)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return None
        
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Draw predictions
        annotated_image = image.copy()
        
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get class and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.CLASSES[cls_id]
            
            # Get color
            color = self.COLORS.get(cls_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
        
        # Save or display
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            print(f"✅ Saved prediction to {output_path}")
        
        return annotated_image, results
    
    def batch_predict(
        self,
        image_dir: str,
        output_dir: str = 'output-Data_Analysis/predictions',
        save: bool = True
    ):
        """
        Run prediction on a directory of images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            save: Whether to save predictions
        """
        image_paths = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.png'))
        
        print(f"Found {len(image_paths)} images")
        
        if save:
            os.makedirs(output_dir, exist_ok=True)
        
        results_list = []
        
        for img_path in image_paths:
            print(f"Processing {img_path.name}...")
            
            if save:
                output_path = os.path.join(output_dir, img_path.name)
            else:
                output_path = None
            
            _, results = self.predict_and_visualize(
                str(img_path),
                output_path=output_path
            )
            results_list.append(results)
        
        print(f"\n✅ Processed {len(image_paths)} images")
        
        return results_list


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='YOLOv11 Inference on BDD100K')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                       help='Image/video path or directory')
    parser.add_argument('--output', type=str, default='output-Data_Analysis/predictions',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save predictions')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Check if source exists
    if not os.path.exists(args.source):
        print(f"❌ Source not found: {args.source}")
        return
    
    # Initialize inference
    inference = BDD100KInference(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Run inference
    if os.path.isdir(args.source):
        print(f"\n📁 Running batch inference on directory: {args.source}\n")
        inference.batch_predict(
            image_dir=args.source,
            output_dir=args.output,
            save=args.save
        )
    else:
        print(f"\n🖼️  Running inference on single image: {args.source}\n")
        output_path = os.path.join(args.output, os.path.basename(args.source))
        inference.predict_and_visualize(
            image_path=args.source,
            output_path=output_path if args.save else None
        )
    
    print("\n🎉 Inference complete!\n")


if __name__ == "__main__":
    main()
