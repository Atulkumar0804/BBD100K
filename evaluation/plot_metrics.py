import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_training_metrics(results_csv_path, output_dir):
    """
    Parses the results.csv from YOLO training and plots separate metrics for Training and Validation.
    """
    if not os.path.exists(results_csv_path):
        print(f"Error: {results_csv_path} not found.")
        return

    # Load data (YOLO CSVs have spaces in headers, strip them)
    df = pd.read_csv(results_csv_path)
    df.columns = [c.strip() for c in df.columns]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot Losses (Train vs Val)
    plt.figure(figsize=(12, 6))
    
    # Box Loss
    plt.subplot(1, 3, 1)
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        plt.title('Box Loss')
        plt.legend()
        plt.grid(True)

    # Cls Loss
    plt.subplot(1, 3, 2)
    if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
        plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
        plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
        plt.title('Class Loss')
        plt.legend()
        plt.grid(True)

    # DFL Loss
    plt.subplot(1, 3, 3)
    if 'train/dfl_loss' in df.columns and 'val/dfl_loss' in df.columns:
        plt.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
        plt.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
        plt.title('Distribution Focal Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_analysis.png'))
    print(f"Saved loss analysis to {output_dir}/loss_analysis.png")
    plt.close()

    # 2. Plot Metrics (mAP, P, R)
    plt.figure(figsize=(10, 5))
    
    # mAP50
    plt.subplot(1, 2, 1)
    if 'metrics/mAP50(B)' in df.columns:
        plt.plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', label='mAP@0.5')
        plt.title('mAP @ 0.5 (Val)')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True)

    # Precision & Recall
    plt.subplot(1, 2, 2)
    if 'metrics/precision(B)' in df.columns:
        plt.plot(df['epoch'], df['metrics/precision(B)'], 'b-', label='Precision')
    if 'metrics/recall(B)' in df.columns:
        plt.plot(df['epoch'], df['metrics/recall(B)'], 'r-', label='Recall')
    plt.title('Precision & Recall (Val)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_analysis.png'))
    print(f"Saved metrics analysis to {output_dir}/metrics_analysis.png")
    plt.close()

if __name__ == "__main__":
    # Find the most recent run
    # Note: Adjust relative path to be correct when run from root
    results_files = glob.glob("bosch-bdd-object-detection/runs-model/train/*/results.csv")
    if results_files:
        latest_results = max(results_files, key=os.path.getctime)
        print(f"Analyzing most recent run: {latest_results}")
        plot_training_metrics(latest_results, "bosch-bdd-object-detection/output-Data_Analysis")
    else:
        results_files_abs = glob.glob("/home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection/runs-model/train/*/results.csv")
        if results_files_abs:
            latest_results = max(results_files_abs, key=os.path.getctime)
            print(f"Analyzing most recent run: {latest_results}")
            plot_training_metrics(latest_results, "bosch-bdd-object-detection/output-Data_Analysis")
        else:
             print("No training results found.")
