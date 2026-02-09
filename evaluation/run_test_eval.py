"""Evaluate YOLOv11 on BDD100K test set with labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Add parent directory to path to resolve imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_analysis.parser import DETECTION_CLASSES, parse_bdd_json
from evaluation.metrics import DetectionMetrics


def run_yolo_test_eval(
    model: YOLO,
    annotations,
    images_dir: Path,
    max_images: int = None,
    device: str = "cpu",
    conf_threshold: float = 0.25,
) -> Dict[str, object]:
    """
    Evaluate YOLOv11 model on test set.
    
    Args:
        model: YOLO model instance
        annotations: List of image annotations
        images_dir: Path to test images directory
        max_images: Maximum number of images to evaluate (None = all)
        device: Device to use ('cuda' or 'cpu')
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = DetectionMetrics(
        num_classes=len(DETECTION_CLASSES),
        class_names=DETECTION_CLASSES
    )

    eval_count = 0
    skip_count = 0
    
    annotations_to_eval = annotations[:max_images] if max_images else annotations
    
    print(f"\nEvaluating on {len(annotations_to_eval)} test images...")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Device: {device}\n")

    for idx, image_ann in enumerate(annotations_to_eval):
        image_path = images_dir / image_ann.image_name
        
        if not image_path.exists():
            print(f"  [{idx+1}/{len(annotations_to_eval)}] SKIP: {image_ann.image_name} (image not found)")
            skip_count += 1
            continue

        try:
            # Load and predict
            image = Image.open(image_path).convert("RGB")
            results = model.predict(
                source=np.array(image),
                device=device,
                conf=conf_threshold,
                verbose=False,
            )
            prediction = results[0]

            # Extract predictions
            boxes = prediction.boxes.xyxy.cpu().numpy().tolist() if prediction.boxes else []
            scores = prediction.boxes.conf.cpu().numpy().tolist() if prediction.boxes else []
            labels = prediction.boxes.cls.cpu().numpy().astype(int).tolist() if prediction.boxes else []

            # Extract ground truth
            gt_boxes = [[obj.x1, obj.y1, obj.x2, obj.y2] for obj in image_ann.objects]
            gt_labels = [DETECTION_CLASSES.index(obj.class_name) for obj in image_ann.objects]

            # Update metrics
            metrics.update(
                [{"boxes": boxes, "labels": labels, "scores": scores}],
                [{"boxes": gt_boxes, "labels": gt_labels}],
            )
            
            eval_count += 1
            
            # Print progress every 50 images
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(annotations_to_eval)}] Processed: {image_ann.image_name}")
                
        except Exception as e:
            print(f"  [{idx+1}/{len(annotations_to_eval)}] ERROR: {image_ann.image_name} - {str(e)}")
            skip_count += 1
            continue

    print(f"\nEvaluation Summary:")
    print(f"  Total images: {len(annotations_to_eval)}")
    print(f"  Successfully evaluated: {eval_count}")
    print(f"  Skipped/Failed: {skip_count}")
    
    return metrics.calculate_all_metrics()


def save_metrics(results: Dict[str, object], output_path: Path) -> None:
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Metrics saved to: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv11 model on BDD100K test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on all test images using best.pt weights
  python3 run_test_eval.py
  
  # Evaluate on first 500 test images
  python3 run_test_eval.py --max_images 500
  
  # Evaluate with custom model and output directory
  python3 run_test_eval.py --yolo_weights path/to/model.pt --output_dir results/
  
  # Evaluate on CPU
  python3 run_test_eval.py --device cpu
  
  # Higher confidence threshold (fewer predictions)
  python3 run_test_eval.py --conf_threshold 0.5
        """
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/bdd100k"),
        help="Path to dataset directory (default: data/bdd100k)"
    )
    parser.add_argument(
        "--test_labels",
        type=Path,
        default=None,
        help="Path to test labels JSON file (default: dataset_dir/labels/bdd100k_labels_images_test.json)"
    )
    parser.add_argument(
        "--test_images",
        type=Path,
        default=None,
        help="Path to test images directory (default: dataset_dir/images/100k/test)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to evaluate (default: None = all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help=f"Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for metrics (default: evaluation/)"
    )
    parser.add_argument(
        "--yolo_weights",
        type=Path,
        default=None,
        help="Path to YOLO weights file (default: runs-model/bdd100k_yolo11_*/weights/best.pt)"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25)"
    )
    
    return parser


def find_latest_model(runs_dir: Path) -> Path:
    """Find the latest trained model."""
    if not runs_dir.exists():
        return None
    
    # Look for best.pt in any run directory
    model_paths = list(runs_dir.glob("**/best.pt"))
    if model_paths:
        # Return the most recently modified one
        return max(model_paths, key=lambda p: p.stat().st_mtime)
    
    return None


def main() -> None:
    """Main evaluation function."""
    args = build_arg_parser().parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Set default paths
    if args.output_dir is None:
        args.output_dir = project_root / "evaluation"
    
    if args.yolo_weights is None:
        # Try to find latest model
        latest_model = find_latest_model(project_root / "runs-model")
        if latest_model:
            args.yolo_weights = latest_model
        else:
            args.yolo_weights = project_root / "runs-model" / "best.pt"
    
    if args.test_labels is None:
        args.test_labels = args.dataset_dir / "labels" / "bdd100k_labels_images_test.json"
    
    if args.test_images is None:
        args.test_images = args.dataset_dir / "images" / "100k" / "test"
    
    # Validate paths
    print("\n" + "="*80)
    print("BDD100K Test Set Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Project root: {project_root}")
    print(f"  Test labels: {args.test_labels}")
    print(f"  Test images: {args.test_images}")
    print(f"  Model weights: {args.yolo_weights}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max images: {args.max_images if args.max_images else 'All'}")
    print(f"  Device: {args.device}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    
    # Check if test labels exist
    if not args.test_labels.exists():
        print(f"\n❌ ERROR: Test labels file not found at {args.test_labels}")
        print(f"\nTo use this script, place test labels at:")
        print(f"  {args.test_labels}")
        print(f"\nExpected format: BDD100K JSON with structure:")
        print(f"""
    [
      {{
        "name": "image_name.jpg",
        "labels": [
          {{
            "category": "car",
            "box2d": {{"x1": 100, "y1": 200, "x2": 300, "y2": 400}},
            "attributes": {{"occluded": false, "truncated": false}}
          }}
        ]
      }}
    ]
        """)
        return
    
    # Check if test images directory exists
    if not args.test_images.exists():
        print(f"\n❌ ERROR: Test images directory not found at {args.test_images}")
        print(f"\nExpected test images at: {args.test_images}")
        return
    
    # Check if model weights exist
    if not args.yolo_weights.exists():
        print(f"\n❌ ERROR: Model weights not found at {args.yolo_weights}")
        print(f"\nTo train a model first, run:")
        print(f"  cd {project_root / 'model'}")
        print(f"  python3 train.py")
        return
    
    # Parse annotations
    print(f"\n📖 Loading test labels from: {args.test_labels}")
    try:
        annotations = parse_bdd_json(args.test_labels)
        print(f"   Loaded {len(annotations)} test images with annotations")
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return
    
    # Load model
    print(f"\n🤖 Loading YOLOv11 model from: {args.yolo_weights}")
    try:
        yolo_model = YOLO(str(args.yolo_weights))
        print(f"   Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run evaluation
    print(f"\n🔍 Starting evaluation...")
    try:
        test_results = run_yolo_test_eval(
            yolo_model,
            annotations,
            args.test_images,
            max_images=args.max_images,
            device=args.device,
            conf_threshold=args.conf_threshold,
        )
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return
    
    # Print results
    print(f"\n📊 Test Set Evaluation Results:")
    print(f"{"="*80}")
    
    if "mAP@0.5" in test_results:
        map_05 = test_results.get("mAP@0.5", {}).get("mAP", 0)
        print(f"  mAP@0.5:     {map_05:.4f}")
    
    if "mAP@0.75" in test_results:
        map_075 = test_results.get("mAP@0.75", {}).get("mAP", 0)
        print(f"  mAP@0.75:    {map_075:.4f}")
    
    if "mAP@[.5:.95]" in test_results:
        map_coco = test_results.get("mAP@[.5:.95]", 0)
        print(f"  mAP@[.5:.95]:{map_coco:.4f} (COCO standard)")
    
    print(f"\nPer-Class Results:")
    if "mAP@0.5" in test_results and "per_class" in test_results["mAP@0.5"]:
        for class_name, class_metrics in test_results["mAP@0.5"]["per_class"].items():
            ap = class_metrics.get("AP", 0)
            precision = class_metrics.get("precision", 0)
            recall = class_metrics.get("recall", 0)
            print(f"  {class_name:15} | AP: {ap:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
    print(f"{"="*80}\n")
    
    # Save metrics
    metrics_root = args.output_dir / "metrics" / "test_set"
    save_metrics(test_results, metrics_root / "metrics.json")
    
    # Save summary
    summary = {
        "test_set_evaluation": True,
        "total_images": len(annotations),
        "max_images_evaluated": args.max_images if args.max_images else len(annotations),
        "device": args.device,
        "confidence_threshold": args.conf_threshold,
        "model_path": str(args.yolo_weights),
        "mAP@0.5": test_results.get("mAP@0.5", {}).get("mAP", 0),
        "mAP@0.75": test_results.get("mAP@0.75", {}).get("mAP", 0),
        "mAP@[.5:.95]": test_results.get("mAP@[.5:.95]", 0),
    }
    save_metrics(summary, metrics_root / "summary.json")
    
    print("✅ Test evaluation completed successfully!")
    print(f"\nResults saved to:")
    print(f"  Detailed metrics: {metrics_root / 'metrics.json'}")
    print(f"  Summary: {metrics_root / 'summary.json'}")


if __name__ == "__main__":
    main()
