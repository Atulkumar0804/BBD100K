"""
Evaluate YOLOv11 model on BDD100K validation set using YOLO's native validation.

This script uses YOLO's built-in validation routine which properly computes mAP metrics
using the official COCO evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# Add parent directory to path to resolve imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_analysis.parser import DETECTION_CLASSES


def save_metrics(results: dict, output_path: Path) -> None:
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv11 model on BDD100K validation set"
    )
    parser.add_argument(
        "--data_yaml",
        type=Path,
        default=Path("configs/bdd100k.yaml"),
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save metrics",
    )
    parser.add_argument(
        "--yolo_weights",
        type=Path,
        default=None,
        help="Path to YOLOv11 weights",
    )
    return parser


def main() -> None:
    """Run evaluation using YOLO's native validation."""
    args = build_arg_parser().parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    
    if args.yolo_weights is None:
        args.yolo_weights = (
            project_root / "runs-model" / 
            "bdd100k_yolo11_20epochs_20260204_234851" / "weights" / "best.pt"
        )
    
    if args.output_dir is None:
        args.output_dir = project_root / "evaluation"
    
    # Make paths absolute
    if not args.yolo_weights.is_absolute():
        args.yolo_weights = project_root / args.yolo_weights
    if not args.data_yaml.is_absolute():
        args.data_yaml = project_root / args.data_yaml
    
    print("=" * 80)
    print("YOLO VALIDATION EVALUATION")
    print("=" * 80)
    print(f"\nModel weights: {args.yolo_weights}")
    print(f"Data YAML: {args.data_yaml}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    
    # Check files exist
    if not args.yolo_weights.exists():
        print(f"\nERROR: Model weights not found at {args.yolo_weights}")
        return
    
    if not args.data_yaml.exists():
        print(f"\nERROR: Data YAML not found at {args.data_yaml}")
        return
    
    # Load model
    print(f"\nLoading model from {args.yolo_weights}...")
    model = YOLO(str(args.yolo_weights))
    
    # Run validation using YOLO's built-in validation
    print(f"\nRunning validation on BDD100K dataset...")
    results = model.val(
        data=str(args.data_yaml),
        imgsz=args.imgsz,
        device=args.device,
        verbose=True,
    )
    
    # Extract metrics
    metrics_data = {
        "mAP@0.5": float(results.box.map50),
        "mAP@0.5:0.95": float(results.box.map),
        "precision": float(results.box.p.mean()),
        "recall": float(results.box.r.mean()),
        "f1": 2 * (float(results.box.p.mean()) * float(results.box.r.mean())) / 
              (float(results.box.p.mean()) + float(results.box.r.mean()) + 1e-6),
        "per_class_metrics": {}
    }
    
    # Add per-class metrics
    if hasattr(results.box, 'ap_class_index'):
        for i, class_name in enumerate(DETECTION_CLASSES):
            if i < len(results.box.ap):
                metrics_data["per_class_metrics"][class_name] = {
                    "AP": float(results.box.ap[i]),
                    "AP50": float(results.box.ap50[i]) if hasattr(results.box, 'ap50') else None,
                    "precision": float(results.box.p[i]) if i < len(results.box.p) else 0,
                    "recall": float(results.box.r[i]) if i < len(results.box.r) else 0,
                }
    
    # Save metrics
    output_path = args.output_dir / "metrics" / "yolov11" / "metrics.json"
    save_metrics(metrics_data, output_path)
    
    # Also save summary
    summary_payload = {
        "metric": "mAP@0.5",
        "scores": {"yolov11": metrics_data["mAP@0.5"]},
        "best_model": "yolov11",
        "best_score": metrics_data["mAP@0.5"],
    }
    save_metrics(summary_payload, args.output_dir / "metrics" / "summary.json")
    
    print(f"\nValidation complete!")
    print(f"\nResults:")
    print(f"   mAP@0.5      : {metrics_data['mAP@0.5']:.4f}")
    print(f"   mAP@0.5:0.95 : {metrics_data['mAP@0.5:0.95']:.4f}")
    print(f"   Precision    : {metrics_data['precision']:.4f}")
    print(f"   Recall       : {metrics_data['recall']:.4f}")
    print(f"   F1 Score     : {metrics_data['f1']:.4f}")
    print(f"\n   Metrics saved to:")
    print(f"     - {output_path}")
    print(f"     - {args.output_dir / 'metrics' / 'summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
