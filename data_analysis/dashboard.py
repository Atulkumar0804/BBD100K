"""BDD100K Streamlit dashboard for analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import streamlit as st


# --- Data Loading & Processing ---

def _load_results(results_path: Path) -> Dict[str, object]:
    """Loads the JSON analysis results."""
    with results_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _class_table(results: Dict[str, object]) -> pd.DataFrame:
    """Formats class distribution data into a DataFrame."""
    class_dist = results["class_distribution"]
    rows = []
    for cls, count in class_dist["train"]["instances"].items():
        rows.append(
            {
                "Class": cls,
                "Train Instances": count,
                "Val Instances": class_dist["val"]["instances"].get(cls, 0),
                "Train Images": class_dist["train"]["images"].get(cls, 0),
                "Val Images": class_dist["val"]["images"].get(cls, 0),
                "Val/Train Inst Ratio": results["split_balance"]["val_to_train_ratio"].get(cls, 0.0),
                "Val/Train Img Ratio": results["split_balance"]["val_to_train_image_ratio"].get(cls, 0.0),
            }
        )
    return pd.DataFrame(rows)


def _bbox_table(results: Dict[str, object]) -> pd.DataFrame:
    """Formats bounding box statistics into a DataFrame."""
    bbox_stats = results["bbox_sizes"]["per_class"]
    rows = [
        {
            "Class": cls,
            "Train Mean": stats["train"]["mean"],
            "Val Mean": stats["val"]["mean"],
            "Train Median": stats["train"]["median"],
            "Val Median": stats["val"]["median"],
        }
        for cls, stats in bbox_stats.items()
    ]
    return pd.DataFrame(rows)


def _iter_images(folder: Path) -> List[Path]:
    """Finds all images in a folder."""
    if not folder.exists():
        return []
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    return sorted(files)


# --- Text Descriptions ---

def _get_plot_description(filename: str) -> str:
    descriptions = {
        "class_distribution.png": "Bar chart comparing class instance counts in train vs val.",
        "class_distribution_pie.png": "Pie charts showing class proportions with totals per split.",
        "val_train_ratio.png": "Ratio of validation to training instances per class.",
        "val_train_image_ratio.png": "Ratio of validation to training image counts per class.",
        "objects_per_image.png": "Histogram of object counts per image for train and val.",
        "objects_per_image_binned.png": "Binned comparison of objects per image (train vs val).",
        "objects_per_image_histogram.png": "Detailed objects-per-image histogram with summary stats.",
        "bbox_size_distribution.png": "Boxplot of bounding box areas per class (train).",
        "bbox_area_histogram.png": "Histogram of bbox areas with mean/median for train and val.",
        "bbox_size_buckets_train.png": "Counts of small/medium/large boxes in train set.",
        "bbox_size_buckets_pie.png": "Pie charts of bbox size buckets for train and val.",
        "avg_bbox_area_per_class.png": "Average bbox area per class (train vs val).",
        "tiny_bbox_per_class.png": "Count of very small boxes per class.",
        "aspect_ratio_scatter.png": "Scatter of bbox width vs height by class.",
        "aspect_ratio_distribution.png": "Distribution of bbox aspect ratios across classes.",
        "class_cooccurrence_matrix.png": "Heatmap of class co-occurrence within images.",
        "spatial_distribution_heatmap.png": "Heatmap of object center locations per class.",
        "cumulative_distributions.png": "CDFs for bbox areas and objects-per-image.",
        "class_imbalance_recommendations.png": "Class imbalance summary with suggested weights.",
    }
    return descriptions.get(filename, "Analysis visualization output.")


def _get_sample_description(filename: str) -> str:
    if filename.startswith("largest_"):
        return "Example image containing the largest bbox for this class."
    if filename.startswith("smallest_"):
        return "Example image containing the smallest bbox for this class."
    if filename == "most_crowded.jpg":
        return "Most crowded image by object count."
    if filename == "rare_class_only.jpg":
        return "Image containing only rare classes."
    return "Sample image with ground-truth boxes."


# --- UI Components ---

def display_gallery(files: List[Path], description_func: callable, cols_per_row: int = 2):
    """
    Displays images in a clean grid layout with consistent caption styling.
    """
    if not files:
        st.info("No images found in this category.")
        return

    # Iterate in batches to create rows
    for i in range(0, len(files), cols_per_row):
        cols = st.columns(cols_per_row)
        batch = files[i : i + cols_per_row]
        
        for idx, path in enumerate(batch):
            desc = description_func(path.name)
            with cols[idx]:
                # Using a container with border creates a 'card' effect
                with st.container():
                    # width='stretch' ensures images fill the column width evenly
                    st.image(str(path), use_column_width=True)
                    
                    # Clean caption formatting
                    st.markdown(f"**{path.name}**")
                    st.caption(desc)


def main() -> None:
    st.set_page_config(
        page_title="BDD100K Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Sidebar ---
    st.sidebar.title("Configuration")
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "output-Data_Analysis"
    
    results_path = st.sidebar.text_input(
        "Analysis Results File", 
        value=str(outputs_dir / "analysis_results.json"),
        help="Path to the JSON file generated by analysis.py"
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Instructions:**\n"
        "1. Ensure `analysis.py` has been run.\n"
        "2. Check that the path to `analysis_results.json` is correct.\n"
        "3. View visual insights in the tabs."
    )

    # --- Main Content ---
    st.title("BDD100K Data Analysis Dashboard")
    st.markdown("Detailed breakdown of class balance, bounding box statistics, dataset anomalies, and visual samples.")

    results_file = Path(results_path)
    if not results_file.exists():
        st.error(
            f"Missing analysis results file at: `{results_path}`\n\n"
            "Please run `analysis.py` first to generate the required data."
        )
        return

    results = _load_results(results_file)
    counts = results.get("image_counts", {})
    
    # Create Tabs for cleaner layout
    tab_overview, tab_tables, tab_plots, tab_samples = st.tabs([
           "Overview & Metrics",
           "Data Tables",
           "Analysis Plots",
           "Sample Images",
    ])

    # --- Tab 1: Overview ---
    with tab_overview:
        st.subheader("Dataset Volume")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train Images (Labels)", counts.get("train_total_labels", results["split_balance"]["total_train_images"]))
        col2.metric("Val Images (Labels)", counts.get("val_total_labels", results["split_balance"]["total_val_images"]))
        col3.metric("Train Images (Disk)", counts.get("train_total_images_disk", "-"))
        col4.metric("Val Images (Disk)", counts.get("val_total_images_disk", "-"))
        
        st.divider()
        
        st.subheader("Quality & Anomalies")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Empty Train Labels", counts.get("train_empty_labels", results["anomalies"]["empty_images"]["train_count"]))
        c2.metric("Empty Val Labels", counts.get("val_empty_labels", results["anomalies"]["empty_images"]["val_count"]))
        c3.metric("Missing Train Labels", counts.get("train_missing_labels", "-"))
        c4.metric("Missing Val Labels", counts.get("val_missing_labels", "-"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Tiny BBoxes Detected", results["anomalies"]["tiny_bboxes"]["count"])
        c6.metric("Classes Missing in Val", len(results["split_balance"]["missing_in_val"]))
        c7.empty()
        c8.empty()

    # --- Tab 2: Data Tables ---
    with tab_tables:
        st.subheader("Class Distribution Analysis")
        st.dataframe(
            _class_table(results), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Val/Train Inst Ratio": st.column_config.NumberColumn(format="%.4f"),
                "Val/Train Img Ratio": st.column_config.NumberColumn(format="%.4f"),
            }
        )

        st.subheader("Bounding Box Statistics")
        st.dataframe(
            _bbox_table(results), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Train Mean": st.column_config.NumberColumn(format="%.1f"),
                "Val Mean": st.column_config.NumberColumn(format="%.1f"),
                "Train Median": st.column_config.NumberColumn(format="%.1f"),
                "Val Median": st.column_config.NumberColumn(format="%.1f"),
            }
        )

    # --- Tab 3: Visualizations ---
    with tab_plots:
        st.subheader("Generated Visualizations")
        plots_dir = outputs_dir / "visualizations"
        st.caption(f"Source Folder: `{plots_dir}`")
        
        plot_files = list(_iter_images(plots_dir))
        display_gallery(plot_files, _get_plot_description, cols_per_row=2)

    # --- Tab 4: Samples ---
    with tab_samples:
        st.subheader("Interesting Sample Images")
        samples_dir = outputs_dir / "interesting_samples"
        st.caption(f"Source Folder: `{samples_dir}`")
        
        sample_files = list(_iter_images(samples_dir))
        display_gallery(sample_files, _get_sample_description, cols_per_row=2)


if __name__ == "__main__":
    main()