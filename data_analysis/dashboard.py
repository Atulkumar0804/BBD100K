"""BDD100K Streamlit dashboard for analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
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


def _attribute_table(results: Dict[str, object], split: str) -> pd.DataFrame:
    """Formats object attributes (occluded/truncated) for a split."""
    data = results["object_attributes"][split]
    rows = []
    for cls, stats in data.items():
        total = stats["total"]
        if total == 0:
            occ_pct = 0.0
            trunc_pct = 0.0
        else:
            occ_pct = (stats["occluded"] / total) * 100
            trunc_pct = (stats["truncated"] / total) * 100
            
        rows.append({
            "Class": cls,
            "Occluded (%)": f"{occ_pct:.1f}%",
            "Truncated (%)": f"{trunc_pct:.1f}%",
            "Total Objects": total
        })
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


def _get_latest_model_run(base_dir: Path) -> Path | None:
    """Finds the most recent model run directory."""
    runs_dir = base_dir / "runs-model"
    if not runs_dir.exists():
        return None
    
    # Get subdirectories
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
        
    # Sort by name (which includes timestamp) or modification time
    subdirs.sort(key=lambda x: x.name, reverse=True)
    return subdirs[0]


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
                st.image(str(path), use_column_width=True)
                st.markdown(f"**{path.name}**")
                st.caption(desc)


# --- Plotting Functions ---

def _plot_classes_interactive(results: Dict[str, object], split: str = "train") -> object:
    """Creates an interactive Plotly bar chart for class distribution."""
    data = results["class_distribution"][split]["instances"]
    df = pd.DataFrame(list(data.items()), columns=["Class", "Instances"])
    # Sort by instances for better visualization
    df = df.sort_values(by="Instances", ascending=False)
    
    fig = px.bar(
        df, 
        x="Class", 
        y="Instances", 
        color="Class",
        title=f"Class Distribution ({split.capitalize()} Set)",
        text="Instances"
    )
    
    # Customize layout: remove numerical text on bars (clean look)
    fig.update_traces(textposition='none') 
    # Ensure legend is visible as requested
    fig.update_layout(showlegend=True)
    return fig


def main() -> None:
    st.set_page_config(
        page_title="BDD100K Analysis Dashboard",
        page_icon="📊",
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
    st.title("📊 BDD100K Data Analysis Dashboard")
    st.markdown("Detailed breakdown of class balance, bounding box statistics, dataset anomalies, and visual samples.")

    results_file = Path(results_path)
    if not results_file.exists():
        st.error(f"❌ Missing analysis results file at: `{results_path}`\n\nPlease run `analysis.py` first to generate the required data.")
        return

    results = _load_results(results_file)
    counts = results.get("image_counts", {})
    
    # Create Tabs for cleaner layout
    tab_overview, tab_tables, tab_plots, tab_attrs, tab_model, tab_samples = st.tabs([
        "📈 Overview & Metrics", 
        "📋 Data Tables", 
        "🎨 Analysis Plots", 
        "🌤️ Attributes",
        "🤖 Model Evaluation",
        "🖼️ Sample Images"
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
            hide_index=True,
            use_container_width=True, 
            column_config={
                "Val/Train Inst Ratio": st.column_config.NumberColumn(format="%.4f"),
                "Val/Train Img Ratio": st.column_config.NumberColumn(format="%.4f"),
            }
        )

        st.subheader("Bounding Box Statistics")
        st.dataframe(
            _bbox_table(results), 
            hide_index=True,
            use_container_width=True, 
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

    # --- Tab 4: Attributes ---
    with tab_attrs:
        st.subheader("Scene Attributes (Weather, Time, Scene)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training Set Scene Stats")
            t_attrs = results["attributes"]["train"]
            st.json(t_attrs)
            
        with col2:
            st.markdown("### Validation Set Scene Stats")
            v_attrs = results["attributes"]["val"]
            st.json(v_attrs)

        st.divider()
        
        st.subheader("Object Attributes (Occlusion & Truncation)")
        st.markdown("Percentage of objects marked as occluded or truncated per class.")
        
        st.markdown("#### Training Set")
        st.dataframe(_attribute_table(results, "train"), use_container_width=True)
        
        st.markdown("#### Validation Set")
        st.dataframe(_attribute_table(results, "val"), use_container_width=True)

    # --- Tab 5: Model Evaluation ---
    with tab_model:
        st.subheader("🎯 Model Training Results (YOLOv11)")
        
        model_run_dir = _get_latest_model_run(outputs_dir)
        
        if not model_run_dir:
            st.warning("⚠️ No model runs found in `output-Data_Analysis/runs-model`.")
            st.info(f"Looking in: `{outputs_dir / 'runs-model'}`")
        else:
            st.success(f"Displaying results for: `{model_run_dir.name}`")
            
            # --- Final Epoch Metrics Table ---
            results_csv_path = model_run_dir / "results.csv"
            if results_csv_path.exists():
                st.markdown("### 🏁 Final Epoch Performance")
                try:
                    df_results = pd.read_csv(results_csv_path)
                    df_results.columns = df_results.columns.str.strip()
                    if not df_results.empty:
                        final_epoch = df_results.iloc[[-1]].copy()
                        cols_to_show = [c for c in final_epoch.columns if "metrics/" in c or "loss" in c or "epoch" in c]
                        final_epoch_display = final_epoch[cols_to_show]
                        st.dataframe(final_epoch_display, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Results CSV is empty.")
                except Exception as e:
                    st.error(f"Error reading results.csv: {e}")
            
            st.divider()

            st.markdown("### 📊 Performance Plots")
            
            # Interactive Class Distribution
            st.markdown("#### Training Class Distribution")
            try:
                fig_dist = _plot_classes_interactive(results, split="train")
                st.plotly_chart(fig_dist)
            except Exception as e:
                st.warning(f"Could not generate interactive plot: {e}")

            # --- Summary Visualizations (Side by Side) ---
            st.markdown("#### Summary Visualizations")
            
            # Labels Correlogram removed as requested
            summary_plots = {
                "Confusion Matrix (Normalized)": "confusion_matrix_normalized.png",
                "Results Summary": "results.png"
            }
            
            s_col1, s_col2 = st.columns(2)
            
            for idx, (title, filename) in enumerate(summary_plots.items()):
                file_path = model_run_dir / filename
                target_col = s_col1 if idx % 2 == 0 else s_col2
                
                with target_col:
                    if file_path.exists():
                        st.image(str(file_path), caption=title)
                        
                        # Add explanation for confusion matrix
                        if filename == "confusion_matrix_normalized.png":
                            with st.expander("ℹ️ Why is 'Background' here?"):
                                st.write("""
                                - **Background (True) vs Class X (Pred):** False Positive.
                                - **Class X (True) vs Background (Pred):** False Negative (Missed object). 
                                """)

            st.divider()
            
            # --- 2x2 Curves Grid (Requested Alignment) ---
            st.markdown("#### Training Metric Curves")
            
            # Row 1: Precision-Recall & F1 Score
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                p_pr = model_run_dir / "BoxPR_curve.png"
                if p_pr.exists():
                    st.image(str(p_pr), caption="Precision-Recall Curve")
                else:
                    st.caption("Precision-Recall Curve not found.")

            with row1_col2:
                p_f1 = model_run_dir / "BoxF1_curve.png"
                if p_f1.exists():
                    st.image(str(p_f1), caption="F1 Score Curve")
                else:
                    st.caption("F1 Score Curve not found.")
            
            # Row 2: Precision & Recall
            row2_col1, row2_col2 = st.columns(2)
            
            with row2_col1:
                p_p = model_run_dir / "BoxP_curve.png"
                if p_p.exists():
                    st.image(str(p_p), caption="Precision Curve")
                else:
                    st.caption("Precision Curve not found.")

            with row2_col2:
                p_r = model_run_dir / "BoxR_curve.png"
                if p_r.exists():
                    st.image(str(p_r), caption="Recall Curve")
                else:
                    st.caption("Recall Curve not found.")

            st.divider()
            
            # --- Training Batches ---
            st.markdown("### 🖼️ Training Batches")
            st.markdown("Samples of training data batches with augmentations.")
            
            train_batches = sorted(list(model_run_dir.glob("train_batch*.jpg")))
            if train_batches:
                cols = st.columns(min(3, len(train_batches)))
                for idx, path in enumerate(train_batches[:3]):
                    with cols[idx % 3]:
                        st.image(str(path), caption=path.name)

            st.divider()
            
            # --- Validation Predictions ---
            st.subheader("🖼️ Validation Predictions")
            st.markdown("Comparing Ground Truth Labels vs Model Predictions.")
            
            val_base_names = set()
            for p in model_run_dir.glob("val_batch*_pred.jpg"):
                val_base_names.add(p.name.replace("_pred.jpg", ""))
            
            if val_base_names:
                sorted_batches = sorted(list(val_base_names))
                for batch_name in sorted_batches:
                    st.markdown(f"**Batch: {batch_name}**")
                    col1, col2 = st.columns(2)
                    label_path = model_run_dir / f"{batch_name}_labels.jpg"
                    pred_path = model_run_dir / f"{batch_name}_pred.jpg"
                    
                    with col1:
                        if label_path.exists():
                            st.image(str(label_path), caption="Ground Truth Labels")
                        else:
                            st.caption("No Label Image")
                    with col2:
                        if pred_path.exists():
                            st.image(str(pred_path), caption="Model Predictions")
                        else:
                            st.caption("No Prediction Image")
                    st.divider()
            else:
                st.info("No validation prediction images found.")

    # --- Tab 6: Samples ---
    with tab_samples:
        st.subheader("Interesting Sample Images")
        samples_dir = outputs_dir / "interesting_samples"
        st.caption(f"Source Folder: `{samples_dir}`")
        
        sample_files = list(_iter_images(samples_dir))
        display_gallery(sample_files, _get_sample_description, cols_per_row=2)


if __name__ == "__main__":
    main()