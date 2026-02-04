"""Evaluate YOLOv11 and torchvision detectors on BDD100K validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from data_analysis.parser import DETECTION_CLASSES, parse_bdd_json
from evaluation.metrics import DetectionMetrics


def run_yolo_eval(
    model: YOLO,
    annotations,
    images_dir: Path,
    max_images: int,
    device: str,
) -> Dict[str, object]:
    metrics = DetectionMetrics(num_classes=len(DETECTION_CLASSES), class_names=DETECTION_CLASSES)

    for idx, image_ann in enumerate(annotations[:max_images]):
        image_path = images_dir / image_ann.image_name
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        results = model.predict(
            source=np.array(image),
            device=device,
            conf=0.25,
            verbose=False,
        )
        prediction = results[0]

        boxes = prediction.boxes.xyxy.cpu().numpy().tolist() if prediction.boxes else []
        scores = prediction.boxes.conf.cpu().numpy().tolist() if prediction.boxes else []
        labels = prediction.boxes.cls.cpu().numpy().astype(int).tolist() if prediction.boxes else []

        gt_boxes = [[obj.x1, obj.y1, obj.x2, obj.y2] for obj in image_ann.objects]
        gt_labels = [DETECTION_CLASSES.index(obj.class_name) for obj in image_ann.objects]

        metrics.update(
            [{"boxes": boxes, "labels": labels, "scores": scores}],
            [{"boxes": gt_boxes, "labels": gt_labels}],
        )

    return metrics.calculate_all_metrics()


def save_metrics(results: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 model")
    parser.add_argument("--dataset_dir", type=Path, default=Path("data"))
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--yolo_weights", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    
    if args.output_dir is None:
        args.output_dir = project_root / "evaluation"
        
    if args.yolo_weights is None:
        args.yolo_weights = project_root / "runs-model" / "train" / "best.pt"

    val_labels = args.dataset_dir / "labels" / "val.json"
    val_images = args.dataset_dir / "images" / "val"

    annotations = parse_bdd_json(val_labels)

    summary = {}
    metrics_root = args.output_dir / "metrics"

    if args.yolo_weights.exists():
        print(f"Evaluating YOLOv11 model: {args.yolo_weights}")
        yolo_model = YOLO(str(args.yolo_weights))
        yolo_results = run_yolo_eval(yolo_model, annotations, val_images, args.max_images, args.device)
        save_metrics(yolo_results, metrics_root / "yolov11" / "metrics.json")
        summary["yolov11"] = yolo_results.get("mAP@0.5", {}).get("mAP")
        print(f"YOLOv11 mAP@0.5: {summary['yolov11']:.4f}")
    else:
        print(f"YOLOv11 weights not found at {args.yolo_weights}")

    if summary:
        summary_payload = {
            "metric": "mAP@0.5",
            "scores": summary,
            "best_model": "yolov11",
            "best_score": summary["yolov11"],
        }
        save_metrics(summary_payload, metrics_root / "summary.json")


if __name__ == "__main__":
    main()
