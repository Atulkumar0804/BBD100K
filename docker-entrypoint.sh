#!/bin/bash
set -e

# Display help
show_help() {
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║          BDD100K Object Detection - Docker Container             ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Available Commands:"
    echo ""
    echo "DATA & ANALYSIS:"
    echo "  analysis          Run data analysis pipeline"
    echo "  dashboard         Launch Streamlit dashboard (port 8501)"
    echo ""
    echo "TRAINING:"
    echo "  train             Train YOLO11 model (requires GPU)"
    echo ""
    echo "INFERENCE & EVALUATION:"
    echo "  inference         Run inference on test data"
    echo "  evaluate          Calculate evaluation metrics"
    echo "  pipeline          Run complete pipeline (analysis → train → eval)"
    echo ""
    echo "INTERACTIVE:"
    echo "  bash              Open bash shell"
    echo "  python            Start Python interpreter"
    echo "  jupyter           Launch Jupyter notebook (port 8888)"
    echo "  tensorboard       Launch TensorBoard (port 6006)"
    echo ""
    echo "SYSTEM:"
    echo "  help              Show this help message"
    echo ""
    echo "Usage: docker run [OPTIONS] bdd100k:latest [COMMAND]"
    echo ""
    echo "Examples:"
    echo "  docker run -it bdd100k:latest analysis"
    echo "  docker run -it --gpus all bdd100k:latest train"
    echo "  docker run -it -p 8501:8501 bdd100k:latest dashboard"
    echo "  docker run -it -p 8888:8888 bdd100k:latest jupyter"
}

# Main command handler
case "${1:-help}" in
    analysis)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ Starting Data Analysis Pipeline..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python /app/data_analysis/parser.py && \
        python /app/data_analysis/analysis.py --output_dir /app/output-Data_Analysis && \
        python /app/data_analysis/visualize.py --output_dir /app/output-Data_Analysis
        echo "✓ Analysis complete!"
        ;;
    
    train|train-yolo)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ Starting YOLO11 Model Training..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python /app/model/train.py --model m --epochs 50 --batch 16
        echo "✓ YOLO11 training complete!"
        ;;
    
    inference)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ Running Inference..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python /app/model/inference.py --model /app/runs-model/train/best.pt --source /app/data
        echo "✓ Inference complete!"
        ;;
    
    evaluate)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ Running Evaluation..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python /app/evaluation/run_model_eval.py
        echo "✓ Evaluation complete!"
        ;;
    
    pipeline)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ Running Complete Pipeline..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Step 1: Data Analysis..."
        python /app/data_analysis/parser.py
        python /app/data_analysis/analysis.py --output_dir /app/output-Data_Analysis
        python /app/data_analysis/visualize.py --output_dir /app/output-Data_Analysis
        echo "✓ Analysis complete!"
        echo ""
        echo "Step 2: Model Training..."
        python /app/model/train.py --model m --epochs 50 --batch 16
        echo "✓ Training complete!"
        echo ""
        echo "Step 3: Inference..."
        python /app/model/inference.py --model /app/runs-model/train/best.pt --source /app/data
        echo "✓ Inference complete!"
        echo ""
        echo "Step 4: Evaluation..."
        python /app/evaluation/run_model_eval.py
        echo "✓ Evaluation complete!"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✓ Complete pipeline finished!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ;;
    
    dashboard)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ Launching Streamlit Dashboard..."
        echo "Dashboard will be available at: http://0.0.0.0:8501"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        streamlit run /app/data_analysis/dashboard.py --server.port 8501 --server.address 0.0.0.0
        ;;
    
    jupyter)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ Launching Jupyter Notebook..."
        echo "Jupyter will be available at: http://0.0.0.0:8888"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
            --NotebookApp.token='' --NotebookApp.password=''
        ;;
    
    tensorboard)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ Launching TensorBoard..."
        echo "TensorBoard will be available at: http://0.0.0.0:6006"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tensorboard --logdir=/app/runs-model --host=0.0.0.0 --port=6006
        ;;
    
    bash)
        exec /bin/bash "${@:2}"
        ;;
    
    python)
        exec python "${@:2}"
        ;;
    
    help)
        show_help
        ;;
    
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
