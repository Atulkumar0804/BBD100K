"""BDD100K visualizations and interesting sample exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

DEFAULT_DPI = 450
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw

from parser import DETECTION_CLASSES, ImageAnnotation, parse_bdd_json

# Color scheme for consistent class visualization
CLASS_COLORS = {
    "person": "#FF6B6B",  # Red
    "rider": "#4ECDC4",  # Teal
    "car": "#45B7D1",  # Blue
    "truck": "#FFA07A",  # Light Salmon
    "bus": "#98D8C8",  # Mint
    "train": "#6C5CE7",  # Purple
    "motor": "#FDCB6E",  # Yellow
    "bike": "#74B9FF",  # Light Blue
    "traffic light": "#55EFC4",  # Aqua
    "traffic sign": "#FD79A8",  # Pink
}


def _add_corner_stats(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.99,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        fontsize=9,
        color="black",
    )


def _load_results(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_index(annotations: List[ImageAnnotation]) -> Dict[str, List[Tuple[str, Tuple[int, int, int, int]]]]:
    index: Dict[str, List[Tuple[str, Tuple[int, int, int, int]]]] = {}
    for image in annotations:
        boxes = []
        for obj in image.objects:
            boxes.append((obj.class_name, (obj.x1, obj.y1, obj.x2, obj.y2)))
        index[image.image_name] = boxes
    return index


def _draw_boxes(image_path: Path, boxes: List[Tuple[str, Tuple[int, int, int, int]]], output_path: Path) -> None:
    if not image_path.exists():
        return

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for class_name, (x1, y1, x2, y2) in boxes:
        color = CLASS_COLORS.get(class_name, "red")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y1 + 2), class_name, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def plot_class_distribution(results: Dict[str, object], output_dir: Path) -> None:
    class_dist = results["class_distribution"]
    classes = DETECTION_CLASSES
    train_counts = [class_dist["train"]["instances"][cls] for cls in classes]
    val_counts = [class_dist["val"]["instances"][cls] for cls in classes]

    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Use class colors for bars
    colors = [CLASS_COLORS.get(cls, '#888888') for cls in classes]
    ax.bar(x - width / 2, train_counts, width, label="Train", color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x + width / 2, val_counts, width, label="Val", color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)

    total_train = sum(train_counts)
    total_val = sum(val_counts)
    stats_text = f"Train total: {total_train:,}\nVal total: {total_val:,}"
    _add_corner_stats(ax, stats_text)
    
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", color="black")
    ax.set_ylabel("Instance Count", fontsize=12, color="black")
    ax.set_title("Class Distribution: Train vs Val (Color-Coded)", fontsize=14, fontweight='bold', color="black")
    ax.legend(fontsize=11, labelcolor="black")
    ax.grid(axis='y', alpha=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_ratio(results: Dict[str, object], output_dir: Path) -> None:
    ratios = results["split_balance"]["val_to_train_ratio"]
    classes = DETECTION_CLASSES
    values = [ratios.get(cls, 0.0) for cls in classes]

    plt.figure(figsize=(12, 6))
    plt.bar(classes, values, color="steelblue")
    plt.xticks(rotation=45, ha="right", color="black")
    plt.ylabel("Val / Train Ratio", color="black")
    plt.title("Train vs Val Ratio per Class", color="black")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "val_train_ratio.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_image_ratio(results: Dict[str, object], output_dir: Path) -> None:
    ratios = results["split_balance"]["val_to_train_image_ratio"]
    classes = DETECTION_CLASSES
    values = [ratios.get(cls, 0.0) for cls in classes]

    plt.figure(figsize=(12, 6))
    plt.bar(classes, values, color="darkorange")
    plt.xticks(rotation=45, ha="right", color="black")
    plt.ylabel("Val / Train Image Ratio", color="black")
    plt.title("Train vs Val Image Ratio per Class", color="black")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "val_train_image_ratio.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_objects_per_image(results: Dict[str, object], output_dir: Path) -> None:
    train_counts = results["objects_per_image"]["train_counts"]
    val_counts = results["objects_per_image"]["val_counts"]

    plt.figure(figsize=(12, 6))
    plt.hist(train_counts, bins=40, alpha=0.7, label="Train")
    plt.hist(val_counts, bins=40, alpha=0.7, label="Val")
    plt.xlabel("Objects per Image", color="black")
    plt.ylabel("Image Count", color="black")
    plt.title("Objects per Image Distribution", color="black")
    plt.legend(labelcolor="black")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "objects_per_image.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_objects_per_image_distribution(results: Dict[str, object], output_dir: Path) -> None:
    dist = results["objects_per_image"]["distribution"]
    bins = list(dist["train"].keys())
    train_values = [dist["train"][b] for b in bins]
    val_values = [dist["val"][b] for b in bins]

    x = np.arange(len(bins))
    width = 0.35
    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, train_values, width, label="Train")
    plt.bar(x + width / 2, val_values, width, label="Val")
    plt.xticks(x, bins, rotation=45, ha="right", color="black")
    plt.ylabel("Image Count", color="black")
    plt.title("Objects per Image (Binned)", color="black")
    plt.legend(labelcolor="black")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "objects_per_image_binned.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_bbox_sizes(annotations: List[ImageAnnotation], output_dir: Path) -> None:
    rows = []
    for image in annotations:
        for obj in image.objects:
            rows.append({"class": obj.class_name, "area": obj.area})

    if not rows:
        return

    df = pd.DataFrame(rows)
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="class", y="area")
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right", color="black")
    plt.title("Bounding Box Area Distribution per Class (Train)", color="black")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bbox_size_distribution.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_bbox_size_buckets(results: Dict[str, object], output_dir: Path) -> None:
    buckets = results["bbox_sizes"]["train"]["size_buckets"]
    labels = list(buckets.keys())
    values = list(buckets.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52"])
    plt.ylabel("Count", color="black")
    plt.title("Bounding Box Size Buckets (Train)", color="black")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bbox_size_buckets_train.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_avg_bbox_area(results: Dict[str, object], output_dir: Path) -> None:
    class_stats = results["class_distribution"]["class_stats"]
    classes = DETECTION_CLASSES
    train_means = [class_stats[cls]["train_avg_bbox_area"] for cls in classes]
    val_means = [class_stats[cls]["val_avg_bbox_area"] for cls in classes]

    x = np.arange(len(classes))
    width = 0.35
    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, train_means, width, label="Train")
    plt.bar(x + width / 2, val_means, width, label="Val")
    plt.xticks(x, classes, rotation=45, ha="right", color="black")
    plt.ylabel("Average BBox Area", color="black")
    plt.title("Average Bounding Box Area per Class", color="black")
    plt.legend(labelcolor="black")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "avg_bbox_area_per_class.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_tiny_bbox_per_class(results: Dict[str, object], output_dir: Path) -> None:
    tiny_stats = results["anomalies"]["tiny_bboxes"]["per_class"]
    classes = DETECTION_CLASSES
    values = [tiny_stats.get(cls, 0) for cls in classes]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(classes, values, color="firebrick")

    total_tiny = sum(values)
    stats_text = f"Total tiny boxes: {total_tiny:,}"
    _add_corner_stats(ax, stats_text)
    
    plt.xticks(rotation=45, ha="right", color="black")
    ax.set_ylabel("Tiny BBox Count", fontsize=12, color="black")
    ax.set_title("Tiny Bounding Boxes per Class", fontsize=14, fontweight='bold', color="black")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "tiny_bbox_per_class.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_class_distribution_pie(results: Dict[str, object], output_dir: Path) -> None:
    """Create pie charts for train and validation class distribution."""
    class_dist = results["class_distribution"]
    classes = DETECTION_CLASSES
    train_counts = [class_dist["train"]["instances"][cls] for cls in classes]
    val_counts = [class_dist["val"]["instances"][cls] for cls in classes]
    
    # Filter out zero values for cleaner pie charts
    train_data = [(cls, count) for cls, count in zip(classes, train_counts) if count > 0]
    val_data = [(cls, count) for cls, count in zip(classes, val_counts) if count > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Train pie chart
    train_labels = [item[0] for item in train_data]
    train_values = [item[1] for item in train_data]
    total_train = sum(train_values)
    
    colors = [CLASS_COLORS.get(cls, '#888888') for cls in train_labels]
    wedges1, _ = ax1.pie(
        train_values,
        labels=None,
        startangle=90,
        colors=colors,
    )
    ax1.legend(
        wedges1,
        [f"{cls}: {count:,}" for cls, count in train_data],
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        title="Train classes",
    )
    _add_corner_stats(ax1, f"Total: {total_train:,}")
    ax1.set_title(f'Train Set Class Distribution\nTotal Instances: {total_train:,}', 
                  fontsize=14, fontweight='bold', pad=20, color="black")
    
    # Val pie chart
    val_labels = [item[0] for item in val_data]
    val_values = [item[1] for item in val_data]
    total_val = sum(val_values)
    
    colors = [CLASS_COLORS.get(cls, '#888888') for cls in val_labels]
    wedges2, _ = ax2.pie(
        val_values,
        labels=None,
        startangle=90,
        colors=colors,
    )
    ax2.legend(
        wedges2,
        [f"{cls}: {count:,}" for cls, count in val_data],
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        title="Val classes",
    )
    _add_corner_stats(ax2, f"Total: {total_val:,}")
    ax2.set_title(f'Validation Set Class Distribution\nTotal Instances: {total_val:,}', 
                  fontsize=14, fontweight='bold', pad=20, color="black")
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "class_distribution_pie.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()


def plot_bbox_size_histogram(train_annotations: List[ImageAnnotation], val_annotations: List[ImageAnnotation], output_dir: Path) -> None:
    """Create histograms for bounding box area distribution."""
    # Collect areas from annotations
    train_areas = []
    for image in train_annotations:
        for obj in image.objects:
            train_areas.append(obj.area)
    
    val_areas = []
    for image in val_annotations:
        for obj in image.objects:
            val_areas.append(obj.area)
    
    if not train_areas or not val_areas:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Train histogram
    ax1.hist(train_areas, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Bounding Box Area (pixels²)', fontsize=12, color="black")
    ax1.set_ylabel('Frequency', fontsize=12, color="black")
    ax1.set_title(f'Train Set: BBox Area Distribution\nTotal: {len(train_areas):,} boxes', 
                  fontsize=14, fontweight='bold', color="black")
    ax1.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    mean_train = np.mean(train_areas)
    median_train = np.median(train_areas)
    ax1.axvline(mean_train, color='red', linestyle='--', linewidth=2, label='Mean')
    ax1.axvline(median_train, color='green', linestyle='--', linewidth=2, label='Median')
    ax1.legend(labelcolor="black")
    _add_corner_stats(ax1, f"Mean: {mean_train:.0f}\nMedian: {median_train:.0f}")
    
    # Val histogram
    ax2.hist(val_areas, bins=50, color='darkorange', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Bounding Box Area (pixels²)', fontsize=12, color="black")
    ax2.set_ylabel('Frequency', fontsize=12, color="black")
    ax2.set_title(f'Validation Set: BBox Area Distribution\nTotal: {len(val_areas):,} boxes', 
                  fontsize=14, fontweight='bold', color="black")
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    mean_val = np.mean(val_areas)
    median_val = np.median(val_areas)
    ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(median_val, color='green', linestyle='--', linewidth=2, label='Median')
    ax2.legend(labelcolor="black")
    _add_corner_stats(ax2, f"Mean: {mean_val:.0f}\nMedian: {median_val:.0f}")
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bbox_area_histogram.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_size_buckets_pie(results: Dict[str, object], output_dir: Path) -> None:
    """Create pie charts for bounding box size buckets."""
    buckets_train = results["bbox_sizes"]["train"]["size_buckets"]
    buckets_val = results["bbox_sizes"]["val"]["size_buckets"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Train size buckets
    labels = list(buckets_train.keys())
    values_train = list(buckets_train.values())
    total_train = sum(values_train)
    colors = ['#4c72b0', '#55a868', '#c44e52']
    
    label_text = [
        f"{l.capitalize()} (<32²)" if l == "small"
        else f"{l.capitalize()} (32²-96²)" if l == "medium"
        else f"{l.capitalize()} (≥96²)" for l in labels
    ]
    wedges1, _ = ax1.pie(
        values_train,
        labels=None,
        colors=colors,
        startangle=90,
    )
    ax1.legend(
        wedges1,
        [f"{lbl}: {val:,}" for lbl, val in zip(label_text, values_train)],
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        title="Train buckets",
    )
    _add_corner_stats(ax1, f"Total: {total_train:,}")
    ax1.set_title(f'Train Set: BBox Size Distribution\nTotal: {total_train:,} boxes', 
                  fontsize=14, fontweight='bold', pad=20, color="black")
    
    # Val size buckets
    values_val = list(buckets_val.values())
    total_val = sum(values_val)
    
    wedges2, _ = ax2.pie(
        values_val,
        labels=None,
        colors=colors,
        startangle=90,
    )
    ax2.legend(
        wedges2,
        [f"{lbl}: {val:,}" for lbl, val in zip(label_text, values_val)],
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        title="Val buckets",
    )
    _add_corner_stats(ax2, f"Total: {total_val:,}")
    ax2.set_title(f'Validation Set: BBox Size Distribution\nTotal: {total_val:,} boxes', 
                  fontsize=14, fontweight='bold', pad=20, color="black")
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bbox_size_buckets_pie.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()


def plot_objects_per_image_detailed(results: Dict[str, object], output_dir: Path) -> None:
    """Create detailed histogram with statistics for objects per image."""
    train_counts = results["objects_per_image"]["train_counts"]
    val_counts = results["objects_per_image"]["val_counts"]

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot histograms
    n_train, bins_train, patches_train = ax.hist(
        train_counts, bins=40, alpha=0.6, label='Train', 
        color='steelblue', edgecolor='black'
    )
    n_val, bins_val, patches_val = ax.hist(
        val_counts, bins=40, alpha=0.6, label='Val', 
        color='darkorange', edgecolor='black'
    )
    
    # Add statistics
    mean_train = np.mean(train_counts)
    mean_val = np.mean(val_counts)
    median_train = np.median(train_counts)
    median_val = np.median(val_counts)
    
    ax.axvline(mean_train, color='blue', linestyle='--', linewidth=2, 
               label='Train Mean')
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label='Val Mean')
    
    ax.set_xlabel('Objects per Image', fontsize=12, color="black")
    ax.set_ylabel('Image Count', fontsize=12, color="black")
    ax.set_title(f'Objects per Image Distribution\nTrain: {len(train_counts):,} images | Val: {len(val_counts):,} images', 
                 fontsize=14, fontweight='bold', color="black")
    ax.legend(fontsize=10, labelcolor="black")
    ax.grid(axis='y', alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Train: min={min(train_counts)}, max={max(train_counts)}, median={median_train:.1f}\n'
    stats_text += f'Val: min={min(val_counts)}, max={max(val_counts)}, median={median_val:.1f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "objects_per_image_histogram.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_aspect_ratio_analysis(train_annotations: List[ImageAnnotation], val_annotations: List[ImageAnnotation], output_dir: Path) -> None:
    """Create aspect ratio analysis plots for better anchor box design."""
    # Collect bbox dimensions per class
    class_data = {cls: {'widths': [], 'heights': [], 'ratios': []} for cls in DETECTION_CLASSES}
    
    for annotation_set in [train_annotations, val_annotations]:
        for image in annotation_set:
            for obj in image.objects:
                width = obj.x2 - obj.x1
                height = obj.y2 - obj.y1
                if height > 0:  # Avoid division by zero
                    ratio = width / height
                    class_data[obj.class_name]['widths'].append(width)
                    class_data[obj.class_name]['heights'].append(height)
                    class_data[obj.class_name]['ratios'].append(ratio)
    
    # Create scatter plot: width vs height
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()
    
    for idx, cls in enumerate(DETECTION_CLASSES):
        ax = axes[idx]
        widths = class_data[cls]['widths']
        heights = class_data[cls]['heights']
        
        if widths and heights:
            ax.scatter(widths, heights, alpha=0.3, s=10, c=CLASS_COLORS.get(cls, '#888888'))
            ax.set_xlabel('Width (px)', fontsize=10, color="black")
            ax.set_ylabel('Height (px)', fontsize=10, color="black")
            ax.set_title(f'{cls.title()}\n({len(widths):,} boxes)', fontsize=11, fontweight='bold', color="black")
            ax.grid(alpha=0.3)
            
            # Add diagonal line for square boxes
            max_val = max(max(widths), max(heights))
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=1, label='Square')
            ax.legend(fontsize=8, labelcolor="black")
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{cls.title()}\n(0 boxes)', fontsize=11)
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "aspect_ratio_scatter.png", dpi=DEFAULT_DPI)
    plt.close()
    
    # Create aspect ratio distribution histogram
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for cls in DETECTION_CLASSES:
        ratios = class_data[cls]['ratios']
        if ratios:
            # Filter extreme outliers for better visualization
            ratios_filtered = [r for r in ratios if 0.1 < r < 10]
            ax.hist(ratios_filtered, bins=50, alpha=0.5, label=cls, color=CLASS_COLORS.get(cls, '#888888'))
    
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Square (1:1)')
    ax.set_xlabel('Aspect Ratio (Width/Height)', fontsize=12, color="black")
    ax.set_ylabel('Frequency', fontsize=12, color="black")
    ax.set_title('Bounding Box Aspect Ratio Distribution (All Classes)', fontsize=14, fontweight='bold', color="black")
    ax.legend(fontsize=9, loc='upper right', labelcolor="black")
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "aspect_ratio_distribution.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_class_cooccurrence_matrix(train_annotations: List[ImageAnnotation], output_dir: Path) -> None:
    """Create heatmap showing which classes appear together in images."""
    # Build co-occurrence matrix
    classes = DETECTION_CLASSES
    n_classes = len(classes)
    cooccurrence = np.zeros((n_classes, n_classes))
    
    for image in train_annotations:
        present_classes = list(set([obj.class_name for obj in image.objects]))
        for i, cls1 in enumerate(classes):
            for j, cls2 in enumerate(classes):
                if cls1 in present_classes and cls2 in present_classes:
                    cooccurrence[i][j] += 1
    
    # Normalize by diagonal (self-occurrence)
    for i in range(n_classes):
        if cooccurrence[i][i] > 0:
            cooccurrence[i] = cooccurrence[i] / cooccurrence[i][i]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(cooccurrence, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=classes, yticklabels=classes, 
                cbar_kws={'label': 'Co-occurrence Probability'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Class Co-occurrence Matrix\n(How often objects appear together)', 
                 fontsize=14, fontweight='bold', pad=20, color="black")
    ax.set_xlabel('Class', fontsize=12, color="black")
    ax.set_ylabel('Class', fontsize=12, color="black")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "class_cooccurrence_matrix.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_spatial_distribution(train_annotations: List[ImageAnnotation], output_dir: Path) -> None:
    """Create heatmap showing where objects appear in images (spatial position bias)."""
    # Create spatial heatmaps for each class
    img_width, img_height = 1280, 720  # BDD100K image dimensions
    grid_size = 20  # 20x20 grid
    
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()
    
    for idx, cls in enumerate(DETECTION_CLASSES):
        ax = axes[idx]
        heatmap = np.zeros((grid_size, grid_size))
        count = 0
        
        for image in train_annotations:
            for obj in image.objects:
                if obj.class_name == cls:
                    # Calculate center point
                    cx = (obj.x1 + obj.x2) / 2
                    cy = (obj.y1 + obj.y2) / 2
                    
                    # Map to grid
                    grid_x = int((cx / img_width) * grid_size)
                    grid_y = int((cy / img_height) * grid_size)
                    
                    # Ensure within bounds
                    grid_x = min(grid_x, grid_size - 1)
                    grid_y = min(grid_y, grid_size - 1)
                    
                    heatmap[grid_y, grid_x] += 1
                    count += 1
        
        if count > 0:
            sns.heatmap(heatmap, cmap='hot', cbar=True, ax=ax, 
                       xticklabels=False, yticklabels=False)
            ax.set_title(f'{cls.title()}\n({count:,} instances)', fontsize=11, fontweight='bold', color="black")
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{cls.title()}\n(0 instances)', fontsize=11)
        
        ax.set_xlabel('Horizontal Position', fontsize=9, color="black")
        ax.set_ylabel('Vertical Position', fontsize=9, color="black")
    
    plt.suptitle('Spatial Distribution: Where Objects Appear in Images', 
                 fontsize=16, fontweight='bold', y=0.995, color="black")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "spatial_distribution_heatmap.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_cumulative_distributions(train_annotations: List[ImageAnnotation], val_annotations: List[ImageAnnotation], output_dir: Path) -> None:
    """Create CDF plots for bbox sizes and objects per image - useful for anchor box design."""
    # Collect bbox areas
    train_areas = []
    val_areas = []
    
    for image in train_annotations:
        for obj in image.objects:
            train_areas.append(obj.area)
    
    for image in val_annotations:
        for obj in image.objects:
            val_areas.append(obj.area)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # CDF of bbox areas
    train_sorted = np.sort(train_areas)
    val_sorted = np.sort(val_areas)
    train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
    val_cdf = np.arange(1, len(val_sorted) + 1) / len(val_sorted)
    
    ax1.plot(train_sorted, train_cdf, label='Train', linewidth=2, color='steelblue')
    ax1.plot(val_sorted, val_cdf, label='Val', linewidth=2, color='darkorange')
    
    # Add percentile lines
    for percentile in [0.5, 0.9, 0.95]:
        train_val = np.percentile(train_areas, percentile * 100)
        ax1.axvline(train_val, color='steelblue', linestyle='--', alpha=0.5)
        ax1.text(train_val, percentile, f'  {percentile*100:.0f}%: {train_val:.0f}px²', 
                fontsize=9, color='steelblue')
    
    ax1.set_xlabel('Bounding Box Area (pixels²)', fontsize=12, color="black")
    ax1.set_ylabel('Cumulative Probability', fontsize=12, color="black")
    ax1.set_title('CDF: Bounding Box Sizes\n(Useful for anchor box design)', 
                  fontsize=14, fontweight='bold', color="black")
    ax1.legend(fontsize=11, labelcolor="black")
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    
    # CDF of objects per image
    train_counts = [len(img.objects) for img in train_annotations]
    val_counts = [len(img.objects) for img in val_annotations]
    
    train_sorted_counts = np.sort(train_counts)
    val_sorted_counts = np.sort(val_counts)
    train_cdf_counts = np.arange(1, len(train_sorted_counts) + 1) / len(train_sorted_counts)
    val_cdf_counts = np.arange(1, len(val_sorted_counts) + 1) / len(val_sorted_counts)
    
    ax2.plot(train_sorted_counts, train_cdf_counts, label='Train', linewidth=2, color='steelblue')
    ax2.plot(val_sorted_counts, val_cdf_counts, label='Val', linewidth=2, color='darkorange')
    
    ax2.set_xlabel('Objects per Image', fontsize=12, color="black")
    ax2.set_ylabel('Cumulative Probability', fontsize=12, color="black")
    ax2.set_title('CDF: Objects per Image Distribution', fontsize=14, fontweight='bold', color="black")
    ax2.legend(fontsize=11, labelcolor="black")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "cumulative_distributions.png", dpi=DEFAULT_DPI)
    plt.close()


def plot_class_imbalance_recommendations(results: Dict[str, object], output_dir: Path) -> None:
    """Create visualization with class imbalance analysis and training recommendations."""
    class_dist = results["class_distribution"]
    classes = DETECTION_CLASSES
    train_counts = [class_dist["train"]["instances"][cls] for cls in classes]
    
    # Calculate imbalance metrics
    max_count = max(train_counts)
    min_count = min([c for c in train_counts if c > 0]) if any(c > 0 for c in train_counts) else 1
    imbalance_ratio = max_count / min_count
    
    # Calculate suggested weights (inverse frequency)
    total = sum(train_counts)
    weights = [total / (len(classes) * max(c, 1)) for c in train_counts]
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main bar chart with colors
    ax1 = fig.add_subplot(gs[0:2, :])
    colors = [CLASS_COLORS.get(cls, '#888888') for cls in classes]
    bars = ax1.bar(classes, train_counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, count in zip(bars, train_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Instance Count', fontsize=14, fontweight='bold', color="black")
    ax1.set_title(f'Class Distribution with Imbalance Analysis\nImbalance Ratio: {imbalance_ratio:.1f}:1', 
                  fontsize=16, fontweight='bold', pad=20, color="black")
    ax1.tick_params(axis='x', rotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add severity indicator
    if imbalance_ratio > 100:
        severity_text = "SEVERE IMBALANCE"
        severity_color = '#FF0000'
    elif imbalance_ratio > 50:
        severity_text = '⚡ MODERATE IMBALANCE'
        severity_color = '#FFA500'
    else:
        severity_text = "BALANCED"
        severity_color = '#00FF00'
    
    ax1.text(0.02, 0.98, severity_text, transform=ax1.transAxes,
            fontsize=14, fontweight='bold', color=severity_color,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=severity_color, linewidth=2))
    
    # Suggested class weights
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.bar(classes, weights, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Suggested Weight', fontsize=12, fontweight='bold', color="black")
    ax2.set_title('Recommended Class Weights for Training', fontsize=13, fontweight='bold', color="black")
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Recommendations text
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axis('off')
    
    recommendations = []
    recommendations.append("TRAINING RECOMMENDATIONS:\n")
    
    if imbalance_ratio > 100:
        recommendations.append("CRITICAL: Severe class imbalance detected!")
        recommendations.append("• Use class weighting (see left chart)")
        recommendations.append("• Apply data augmentation for rare classes")
        recommendations.append("• Consider focal loss or OHEM")
        recommendations.append("• Use stratified sampling")
    elif imbalance_ratio > 50:
        recommendations.append("⚡ MODERATE: Noticeable imbalance")
        recommendations.append("• Apply class weights during training")
        recommendations.append("• Monitor per-class metrics closely")
        recommendations.append("• Consider oversampling rare classes")
    else:
        recommendations.append("GOOD: Relatively balanced dataset")
        recommendations.append("• Standard training should work well")
        recommendations.append("• Monitor all classes equally")
    
    recommendations.append("\nSTATISTICS:")
    recommendations.append(f"• Most common: {classes[train_counts.index(max_count)]} ({max_count:,})")
    recommendations.append(f"• Least common: {classes[train_counts.index(min_count)]} ({min_count:,})")
    recommendations.append(f"• Ratio: {imbalance_ratio:.1f}:1")
    
    rec_text = '\n'.join(recommendations)
    ax3.text(0.05, 0.95, rec_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=2))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "class_imbalance_recommendations.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()


def export_interesting_samples(
    train_images_dir: Path,
    annotations: List[ImageAnnotation],
    results: Dict[str, object],
    output_dir: Path,
) -> None:
    index = _build_index(annotations)
    samples = results.get("interesting_samples", {})

    for cls, info in samples.get("largest_bbox_per_class", {}).items():
        image_name = info["image"]
        boxes = index.get(image_name, [])
        _draw_boxes(
            train_images_dir / image_name,
            boxes,
            output_dir / f"largest_{cls.replace(' ', '_')}.jpg",
        )

    for cls, info in samples.get("smallest_bbox_per_class", {}).items():
        image_name = info["image"]
        boxes = index.get(image_name, [])
        _draw_boxes(
            train_images_dir / image_name,
            boxes,
            output_dir / f"smallest_{cls.replace(' ', '_')}.jpg",
        )

    crowded = samples.get("most_crowded_image", {}).get("image")
    if crowded:
        _draw_boxes(
            train_images_dir / crowded,
            index.get(crowded, []),
            output_dir / "most_crowded.jpg",
        )

    rare_only = samples.get("rare_class_only_image")
    if rare_only:
        _draw_boxes(
            train_images_dir / rare_only,
            index.get(rare_only, []),
            output_dir / "rare_class_only.jpg",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BDD100K visualization")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data"),
        help="Root dataset directory (contains images/ and labels/)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output-Data_Analysis"),
        help="Output directory for plots and samples",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_path = args.output_dir / "analysis_results.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing analysis results at {results_path}. Run analysis.py first."
        )

    results = _load_results(results_path)
    train_labels = Path(results["dataset"]["train_labels"])
    val_labels = Path(results["dataset"]["val_labels"])
    train_images_dir = Path(results["dataset"]["train_images"])

    train_annotations = parse_bdd_json(train_labels)
    val_annotations = parse_bdd_json(val_labels)

    vis_dir = args.output_dir / "visualizations"
    
    print("\nGenerating visualizations...")
    print("  ├─ Class distribution charts...")
    plot_class_distribution(results, vis_dir)
    plot_class_distribution_pie(results, vis_dir)
    
    print("  ├─ Train/Val ratio analysis...")
    plot_ratio(results, vis_dir)
    plot_image_ratio(results, vis_dir)
    
    print("  ├─ Objects per image statistics...")
    plot_objects_per_image(results, vis_dir)
    plot_objects_per_image_distribution(results, vis_dir)
    plot_objects_per_image_detailed(results, vis_dir)
    
    print("  ├─ Bounding box analysis...")
    plot_bbox_sizes(train_annotations, vis_dir)
    plot_bbox_size_buckets(results, vis_dir)
    plot_bbox_size_histogram(train_annotations, val_annotations, vis_dir)
    plot_size_buckets_pie(results, vis_dir)
    plot_avg_bbox_area(results, vis_dir)
    plot_tiny_bbox_per_class(results, vis_dir)
    
    print("  ├─ NEW: Aspect ratio analysis...")
    plot_aspect_ratio_analysis(train_annotations, val_annotations, vis_dir)
    
    print("  ├─ NEW: Class co-occurrence matrix...")
    plot_class_cooccurrence_matrix(train_annotations, vis_dir)
    
    print("  ├─ NEW: Spatial distribution heatmaps...")
    plot_spatial_distribution(train_annotations, vis_dir)
    
    print("  ├─ NEW: Cumulative distributions (CDF)...")
    plot_cumulative_distributions(train_annotations, val_annotations, vis_dir)
    
    print("  ├─ NEW: Class imbalance recommendations...")
    plot_class_imbalance_recommendations(results, vis_dir)
    
    print("  └─ Exporting interesting samples...")
    export_interesting_samples(
        train_images_dir,
        train_annotations,
        results,
        args.output_dir / "interesting_samples",
    )
    
    # Count total visualizations
    total_plots = len(list(vis_dir.glob('*.png')))
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Generated {total_plots} high-quality visualization plots")
    print(f"Location: {vis_dir}")
    print("\nNEW FEATURES ADDED:")
    print("  - Color-coded class visualizations")
    print("  - Aspect ratio analysis (for anchor box design)")
    print("  - Class co-occurrence matrix")
    print("  - Spatial distribution heatmaps")
    print("  - Cumulative distribution functions (CDF)")
    print("  - Class imbalance recommendations")
    print("  - Enhanced bbox colors in sample images")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
