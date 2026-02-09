"""BDD100K data analysis pipeline for detection labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parser import (
    DETECTION_CLASSES,
    BoundingBox,
    ImageAnnotation,
    iter_all_objects,
    parse_bdd_json,
)


def resolve_dataset_paths(dataset_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Resolve dataset paths using the BDD100K structure."""
    
    # Auto-detect subdirectory if data is inside 'bdd100k'
    if (dataset_dir / "bdd100k").exists() and (dataset_dir / "bdd100k" / "labels").exists():
        dataset_dir = dataset_dir / "bdd100k"
    
    # Try standard BDD structure first
    train_labels = dataset_dir / "labels" / "bdd100k_labels_images_train.json"
    val_labels = dataset_dir / "labels" / "bdd100k_labels_images_val.json"
    
    # Fallback to simple structure if BDD specific files don't exist
    if not train_labels.exists():
        train_labels = dataset_dir / "labels" / "train.json"
    if not val_labels.exists():
        val_labels = dataset_dir / "labels" / "val.json"

    train_images = dataset_dir / "images" / "100k" / "train"
    val_images = dataset_dir / "images" / "100k" / "val"
    
    # Fallback for images if strict bdd structure missing
    if not train_images.exists():
        train_images = dataset_dir / "images" / "train"
    if not val_images.exists():
        val_images = dataset_dir / "images" / "val"

    return train_labels, val_labels, train_images, val_images


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _count_images_in_dir(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return sum(
        1
        for name in os.listdir(images_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _list_images_in_dir(images_dir: Path) -> List[str]:
    if not images_dir.exists():
        return []
    return sorted(
        name
        for name in os.listdir(images_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _count_empty_in_labels(label_path: Path) -> int:
    with label_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    empty = 0
    for entry in raw:
        labels = entry.get("labels", []) or []
        if not labels:
            empty += 1
            continue
        has_valid = False
        for label in labels:
            box = label.get("box2d")
            if box:
                has_valid = True
                break
        if not has_valid:
            empty += 1
    return empty


def _class_instance_counts(annotations: Iterable[ImageAnnotation]) -> Dict[str, int]:
    counts = {cls: 0 for cls in DETECTION_CLASSES}
    for obj in iter_all_objects(annotations):
        counts[obj.class_name] += 1
    return counts


def _class_image_counts(annotations: Iterable[ImageAnnotation]) -> Dict[str, int]:
    counts = {cls: 0 for cls in DETECTION_CLASSES}
    for image in annotations:
        present = {obj.class_name for obj in image.objects}
        for cls in present:
            counts[cls] += 1
    return counts


def _bbox_areas_by_class(annotations: Iterable[ImageAnnotation]) -> Dict[str, List[int]]:
    areas = {cls: [] for cls in DETECTION_CLASSES}
    for obj in iter_all_objects(annotations):
        areas[obj.class_name].append(obj.area)
    return areas


def _attribute_counts(annotations: Iterable[ImageAnnotation]) -> Dict[str, Dict[str, int]]:
    """Count image-level attributes (weather, scene, timeofday)."""
    counts = {
        "weather": {},
        "scene": {},
        "timeofday": {}
    }
    for img in annotations:
        counts["weather"][img.weather] = counts["weather"].get(img.weather, 0) + 1
        counts["scene"][img.scene] = counts["scene"].get(img.scene, 0) + 1
        counts["timeofday"][img.timeofday] = counts["timeofday"].get(img.timeofday, 0) + 1
    return counts


def _object_attribute_counts(annotations: Iterable[ImageAnnotation]) -> Dict[str, Dict[str, int]]:
    """Count object-level attributes (occluded, truncated) per class."""
    stats = {cls: {"occluded": 0, "truncated": 0, "total": 0} for cls in DETECTION_CLASSES}
    
    for obj in iter_all_objects(annotations):
        stats[obj.class_name]["total"] += 1
        if obj.occluded:
            stats[obj.class_name]["occluded"] += 1
        if obj.truncated:
            stats[obj.class_name]["truncated"] += 1
            
    return stats


def _describe(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _percentiles(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _size_buckets(areas: List[int]) -> Dict[str, int]:
    small = sum(1 for a in areas if a < 32 * 32)
    medium = sum(1 for a in areas if 32 * 32 <= a < 96 * 96)
    large = sum(1 for a in areas if a >= 96 * 96)
    return {"small": small, "medium": medium, "large": large}


def _objects_per_image_distribution(values: List[int], max_bin: int = 20) -> Dict[str, int]:
    buckets = {str(i): 0 for i in range(max_bin + 1)}
    buckets[">20"] = 0
    for value in values:
        if value > max_bin:
            buckets[">20"] += 1
        else:
            buckets[str(value)] += 1
    return buckets


def _bbox_dict(obj: BoundingBox) -> Dict[str, int | str]:
    return {
        "class_name": obj.class_name,
        "x1": obj.x1,
        "y1": obj.y1,
        "x2": obj.x2,
        "y2": obj.y2,
        "area": obj.area,
    }


class BDD100KAnalyzer:
    """Analysis pipeline for the BDD100K object detection dataset."""

    def __init__(self, train_labels: Path, val_labels: Path) -> None:
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.train_data: List[ImageAnnotation] = []
        self.val_data: List[ImageAnnotation] = []
        self.results: Dict[str, object] = {}

    def load(self) -> None:
        """Load and parse train/val JSON labels into dataclasses."""

        self.train_data = parse_bdd_json(self.train_labels)
        self.val_data = parse_bdd_json(self.val_labels)

    def analyze_class_distribution(self) -> Dict[str, object]:
        """Compute instances and images per class for train/val."""

        train_instances = _class_instance_counts(self.train_data)
        val_instances = _class_instance_counts(self.val_data)
        train_images = _class_image_counts(self.train_data)
        val_images = _class_image_counts(self.val_data)

        total_train = sum(train_instances.values())
        total_val = sum(val_instances.values())

        train_percent = {
            cls: (train_instances[cls] / total_train * 100) if total_train else 0.0
            for cls in DETECTION_CLASSES
        }
        val_percent = {
            cls: (val_instances[cls] / total_val * 100) if total_val else 0.0
            for cls in DETECTION_CLASSES
        }

        class_stats = {}
        train_areas = _bbox_areas_by_class(self.train_data)
        val_areas = _bbox_areas_by_class(self.val_data)
        for cls in DETECTION_CLASSES:
            class_stats[cls] = {
                "train_avg_bbox_area": _describe(train_areas[cls])["mean"],
                "val_avg_bbox_area": _describe(val_areas[cls])["mean"],
                "train_images_with_class": train_images[cls],
                "val_images_with_class": val_images[cls],
                "train_avg_objects_per_image": (
                    train_instances[cls] / train_images[cls] if train_images[cls] else 0.0
                ),
                "val_avg_objects_per_image": (
                    val_instances[cls] / val_images[cls] if val_images[cls] else 0.0
                ),
            }

        analysis = {
            "train": {"instances": train_instances, "images": train_images},
            "val": {"instances": val_instances, "images": val_images},
            "train_percent": train_percent,
            "val_percent": val_percent,
            "total_train_instances": total_train,
            "total_val_instances": total_val,
            "class_stats": class_stats,
        }

        self.results["class_distribution"] = analysis
        return analysis

    def analyze_objects_per_image(self) -> Dict[str, object]:
        """Compute objects-per-image distributions for train/val."""

        train_counts = [len(img.objects) for img in self.train_data]
        val_counts = [len(img.objects) for img in self.val_data]

        analysis = {
            "train": {**_describe(train_counts), **_percentiles(train_counts)},
            "val": {**_describe(val_counts), **_percentiles(val_counts)},
            "train_counts": train_counts,
            "val_counts": val_counts,
            "distribution": {
                "train": _objects_per_image_distribution(train_counts),
                "val": _objects_per_image_distribution(val_counts),
            },
        }

        self.results["objects_per_image"] = analysis
        return analysis

    def analyze_bbox_sizes(self) -> Dict[str, object]:
        """Analyze bounding box size distributions."""

        train_areas = [obj.area for obj in iter_all_objects(self.train_data)]
        val_areas = [obj.area for obj in iter_all_objects(self.val_data)]

        per_class_train = _bbox_areas_by_class(self.train_data)
        per_class_val = _bbox_areas_by_class(self.val_data)

        per_class_stats = {}
        for cls in DETECTION_CLASSES:
            per_class_stats[cls] = {
                "train": _describe(per_class_train[cls]),
                "val": _describe(per_class_val[cls]),
                "train_size_buckets": _size_buckets(per_class_train[cls]),
                "val_size_buckets": _size_buckets(per_class_val[cls]),
            }

        analysis = {
            "train": {
                **_describe(train_areas),
                "size_buckets": _size_buckets(train_areas),
            },
            "val": {
                **_describe(val_areas),
                "size_buckets": _size_buckets(val_areas),
            },
            "per_class": per_class_stats,
        }

        self.results["bbox_sizes"] = analysis
        return analysis

    def analyze_split_balance(self) -> Dict[str, object]:
        """Compare train/val class balance with ratios."""

        class_dist = self.results.get("class_distribution") or self.analyze_class_distribution()
        train_counts = class_dist["train"]["instances"]
        val_counts = class_dist["val"]["instances"]

        val_to_train_ratio = {}
        val_to_train_image_ratio = {}
        for cls in DETECTION_CLASSES:
            train_val = train_counts.get(cls, 0)
            val_val = val_counts.get(cls, 0)
            val_to_train_ratio[cls] = (val_val / train_val) if train_val else 0.0

            train_imgs = class_dist["train"]["images"].get(cls, 0)
            val_imgs = class_dist["val"]["images"].get(cls, 0)
            val_to_train_image_ratio[cls] = (val_imgs / train_imgs) if train_imgs else 0.0

        missing_in_val = [cls for cls in DETECTION_CLASSES if val_counts.get(cls, 0) == 0]

        bbox_sizes = self.results.get("bbox_sizes") or self.analyze_bbox_sizes()
        bbox_shift_ratio = {
            cls: (
                bbox_sizes["per_class"][cls]["val"]["mean"]
                / bbox_sizes["per_class"][cls]["train"]["mean"]
                if bbox_sizes["per_class"][cls]["train"]["mean"]
                else 0.0
            )
            for cls in DETECTION_CLASSES
        }

        analysis = {
            "val_to_train_ratio": val_to_train_ratio,
            "val_to_train_image_ratio": val_to_train_image_ratio,
            "bbox_mean_shift_ratio": bbox_shift_ratio,
            "missing_in_val": missing_in_val,
            "total_train_images": len(self.train_data),
            "total_val_images": len(self.val_data),
        }

        self.results["split_balance"] = analysis
        return analysis

    def find_anomalies(self) -> Dict[str, object]:
        """Identify anomalies like empty images, tiny boxes, crowded scenes."""

        empty_train = [img.image_name for img in self.train_data if not img.objects]
        empty_val = [img.image_name for img in self.val_data if not img.objects]

        tiny_box_area_ratio = 0.001
        tiny_boxes = []
        tiny_boxes_per_class = {cls: 0 for cls in DETECTION_CLASSES}
        for image in self.train_data:
            image_area = max(1, image.width * image.height)
            tiny_threshold = max(1, int(tiny_box_area_ratio * image_area))
            for obj in image.objects:
                if obj.area < tiny_threshold:
                    tiny_boxes.append(
                        {
                            "image": image.image_name,
                            "bbox": _bbox_dict(obj),
                            "area": obj.area,
                        }
                    )
                    tiny_boxes_per_class[obj.class_name] += 1

        sorted_by_count = sorted(
            self.train_data, key=lambda img: len(img.objects), reverse=True
        )
        top_k = max(1, int(len(sorted_by_count) * 0.01)) if sorted_by_count else 0
        crowded = [
            {"image": img.image_name, "object_count": len(img.objects)}
            for img in sorted_by_count[:top_k]
        ]

        anomalies = {
            "empty_images": {
                "train_count": len(empty_train),
                "val_count": len(empty_val),
                "train_examples": empty_train[:10],
            },
            "tiny_bboxes": {
                "threshold_area": int(tiny_box_area_ratio * 1280 * 720),
                "threshold_ratio": tiny_box_area_ratio,
                "count": len(tiny_boxes),
                "per_class": tiny_boxes_per_class,
                "examples": tiny_boxes[:10],
            },
            "crowded_images": {
                "top_fraction": 0.01,
                "count": len(crowded),
                "examples": crowded[:10],
            },
        }

        self.results["anomalies"] = anomalies
        return anomalies

    def select_interesting_samples(self) -> Dict[str, object]:
        """Select interesting samples for visualization."""

        largest_per_class = {}
        smallest_per_class = {}

        for image in self.train_data:
            for obj in image.objects:
                cls = obj.class_name
                current_large = largest_per_class.get(cls)
                if current_large is None or obj.area > current_large["bbox"].area:
                    largest_per_class[cls] = {"image": image.image_name, "bbox": obj}

                current_small = smallest_per_class.get(cls)
                if current_small is None or obj.area < current_small["bbox"].area:
                    smallest_per_class[cls] = {"image": image.image_name, "bbox": obj}

        class_dist = self.results.get("class_distribution") or self.analyze_class_distribution()
        train_instances = class_dist["train"]["instances"]
        rare_classes = sorted(train_instances, key=train_instances.get)[:3]

        rare_only_image: Optional[str] = None
        for image in self.train_data:
            if not image.objects:
                continue
            classes = {obj.class_name for obj in image.objects}
            if classes.issubset(set(rare_classes)):
                rare_only_image = image.image_name
                break

        most_crowded = max(self.train_data, key=lambda img: len(img.objects), default=None)

        samples = {
            "largest_bbox_per_class": {
                cls: {
                    "image": data["image"],
                    "bbox": _bbox_dict(data["bbox"]),
                }
                for cls, data in largest_per_class.items()
            },
            "smallest_bbox_per_class": {
                cls: {
                    "image": data["image"],
                    "bbox": _bbox_dict(data["bbox"]),
                }
                for cls, data in smallest_per_class.items()
            },
            "most_crowded_image": {
                "image": most_crowded.image_name if most_crowded else None,
                "object_count": len(most_crowded.objects) if most_crowded else 0,
            },
            "rare_class_only_image": rare_only_image,
            "rare_classes": rare_classes,
        }

        self.results["interesting_samples"] = samples
        return samples

    def run(self) -> Dict[str, object]:
        """Execute the full analysis pipeline."""

        if not self.train_data or not self.val_data:
            self.load()

        # Run Analysis
        print(" Analyzing Class Distribution...")
        train_counts = _class_instance_counts(self.train_data)
        val_counts = _class_instance_counts(self.val_data)

        print(" Analyzing Image Attributes...")
        train_attrs = _attribute_counts(self.train_data)
        val_attrs = _attribute_counts(self.val_data)
        
        print(" Analyzing Object Attributes...")
        train_obj_attrs = _object_attribute_counts(self.train_data)
        val_obj_attrs = _object_attribute_counts(self.val_data)

        print(" Analyzing Bounding Boxes...")
        train_areas = _bbox_areas_by_class(self.train_data)
        val_areas = _bbox_areas_by_class(self.val_data)

        bbox_stats = {}
        for cls in DETECTION_CLASSES:
            bbox_stats[cls] = {
                "train": _describe(train_areas[cls]),
                "val": _describe(val_areas[cls]),
            }

        bbox_buckets_train = _size_buckets([area for areas in train_areas.values() for area in areas])
        bbox_buckets_val = _size_buckets([area for areas in val_areas.values() for area in areas])

        # Restore missing calculations
        train_img_counts = _class_image_counts(self.train_data)
        val_img_counts = _class_image_counts(self.val_data)
        
        split_balance = {
             "total_train_images": len(self.train_data),
             "total_val_images": len(self.val_data),
             "val_to_train_ratio": {},
             "val_to_train_image_ratio": {},
             "missing_in_val": []
        }
        
        for cls in DETECTION_CLASSES:
            train_inst = train_counts.get(cls, 0)
            val_inst = val_counts.get(cls, 0)
            ratio_inst = val_inst / train_inst if train_inst > 0 else 0.0
            split_balance["val_to_train_ratio"][cls] = ratio_inst
            
            train_img = train_img_counts.get(cls, 0)
            val_img = val_img_counts.get(cls, 0)
            ratio_img = val_img / train_img if train_img > 0 else 0.0
            split_balance["val_to_train_image_ratio"][cls] = ratio_img
            
            if val_inst == 0 and train_inst > 0:
                split_balance["missing_in_val"].append(cls)

        # Anomalies
        tiny_count = 0
        tiny_per_class = {cls: 0 for cls in DETECTION_CLASSES}
        
        for ann in self.train_data + self.val_data:
            for obj in ann.objects:
                if obj.area < 50: # Arbitrary tiny threshold
                    tiny_count += 1
                    tiny_per_class[obj.class_name] += 1
                    
        anomalies = {
             "empty_images": {
                  "train_count": sum(1 for img in self.train_data if not img.objects),
                  "val_count": sum(1 for img in self.val_data if not img.objects)
             },
             "tiny_bboxes": {
                  "count": tiny_count,
                  "per_class": tiny_per_class
             }
        }
        
        # New calculation for objects per image
        train_obj_counts = [len(img.objects) for img in self.train_data]
        val_obj_counts = [len(img.objects) for img in self.val_data]
        
        from collections import Counter
        objects_per_image = {
             "train_counts": train_obj_counts,
             "val_counts": val_obj_counts,
             "distribution": {
                 "train": dict(Counter(train_obj_counts)),
                 "val": dict(Counter(val_obj_counts))
             }
        }

        analysis = {
            "class_distribution": {
                "train": {
                    "instances": train_counts,
                    "images": train_img_counts,
                },
                "val": {
                    "instances": val_counts,
                    "images": val_img_counts,
                },
            },
            "split_balance": split_balance,
            "anomalies": anomalies,
            "objects_per_image": objects_per_image,
            "attributes": {
                "train": train_attrs,
                "val": val_attrs
            },
            "object_attributes": {
                "train": train_obj_attrs,
                "val": val_obj_attrs
            },
            "bbox_sizes": {
                "per_class": bbox_stats,
                "buckets": {
                    "train": bbox_buckets_train,
                    "val": bbox_buckets_val,
                },
            },
        }

        self.results = analysis
        
        print(" Selecting Interesting Samples...")
        self.select_interesting_samples()
        
        return self.results


def save_results(results: Dict[str, object], output_path: Path) -> None:
    """Save results to JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(convert(results), handle, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="BDD100K data analysis")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/bdd100k"),
        help="Root dataset directory (contains images/ and labels/)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output-Data_Analysis"),
        help="Output directory for JSON results",
    )
    return parser


def main() -> None:
    """CLI entrypoint for analysis."""

    args = build_arg_parser().parse_args()
    train_labels, val_labels, train_images, val_images = resolve_dataset_paths(args.dataset_dir)

    _ensure_exists(train_labels, "train labels")
    _ensure_exists(val_labels, "val labels")
    _ensure_exists(train_images, "train images directory")
    _ensure_exists(val_images, "val images directory")

    analyzer = BDD100KAnalyzer(train_labels, val_labels)
    results = analyzer.run()

    with train_labels.open("r", encoding="utf-8") as handle:
        raw_train = json.load(handle)
    with val_labels.open("r", encoding="utf-8") as handle:
        raw_val = json.load(handle)

    train_label_names = {entry.get("name") for entry in raw_train if entry.get("name")}
    val_label_names = {entry.get("name") for entry in raw_val if entry.get("name")}
    train_disk_names = _list_images_in_dir(train_images)
    val_disk_names = _list_images_in_dir(val_images)

    missing_train_labels = [name for name in train_disk_names if name not in train_label_names]
    missing_val_labels = [name for name in val_disk_names if name not in val_label_names]

    results["image_counts"] = {
        "train_total_labels": len(raw_train),
        "val_total_labels": len(raw_val),
        "train_total_images_disk": _count_images_in_dir(train_images),
        "val_total_images_disk": _count_images_in_dir(val_images),
        "train_empty_labels": _count_empty_in_labels(train_labels),
        "val_empty_labels": _count_empty_in_labels(val_labels),
        "train_missing_labels": len(missing_train_labels),
        "val_missing_labels": len(missing_val_labels),
        "train_missing_label_examples": missing_train_labels[:10],
        "val_missing_label_examples": missing_val_labels[:10],
    }

    results["dataset"] = {
        "dataset_dir": str(args.dataset_dir),
        "train_labels": str(train_labels),
        "val_labels": str(val_labels),
        "train_images": str(train_images),
        "val_images": str(val_images),
    }

    save_results(results, args.output_dir / "analysis_results.json")


if __name__ == "__main__":
    main()
