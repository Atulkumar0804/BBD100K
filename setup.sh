#!/bin/bash

# Setup script for BDD100K Object Detection Project
# This script creates necessary directories and checks dependencies

echo "=========================================="
echo "BDD100K Object Detection - Setup Script"
echo "=========================================="
echo ""

# Create directory structure
echo "📁 Creating directory structure..."

mkdir -p data/bdd100k/images/100k/train
mkdir -p data/bdd100k/images/100k/val
mkdir -p data/bdd100k/labels/det_20
mkdir -p data/parsed
mkdir -p output-Data_Analysis/visualizations
mkdir -p output-Data_Analysis/predictions
mkdir -p output-Data_Analysis/error_analysis
mkdir -p runs-model/train
mkdir -p configs
mkdir -p logs

echo "✅ Directories created"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "✅ Python 3.9+ detected"
else
    echo "❌ Python 3.9+ required. Please upgrade Python."
    exit 1
fi
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Check for CUDA
echo "🎮 Checking for CUDA..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")
    echo "✅ CUDA available (version: $cuda_version)"
else
    echo "⚠️  CUDA not available. Training will use CPU (very slow)."
    echo "   For GPU training, install CUDA and PyTorch with CUDA support."
fi
echo ""

# Check Docker
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version | awk '{print $3}')
    echo "✅ Docker installed (version: $docker_version)"
    
    if command -v docker-compose &> /dev/null; then
        compose_version=$(docker-compose --version | awk '{print $3}')
        echo "✅ Docker Compose installed (version: $compose_version)"
    else
        echo "⚠️  Docker Compose not found. Install for easy container orchestration."
    fi
else
    echo "⚠️  Docker not found. Install Docker for containerized workflows."
fi
echo ""

# Check dataset
echo "📊 Checking for BDD100K dataset..."
if [ -f "data/bdd100k/labels/det_20/det_train.json" ]; then
    echo "✅ Training annotations found"
else
    echo "❌ Training annotations not found at: data/bdd100k/labels/det_20/det_train.json"
    echo ""
    echo "📥 Please download BDD100K dataset:"
    echo "   1. Visit: https://bdd-data.berkeley.edu/"
    echo "   2. Download '100K Images' and 'Detection 2020 Labels'"
    echo "   3. Extract to: data/bdd100k/"
    echo ""
fi

if [ -f "data/bdd100k/labels/det_20/det_val.json" ]; then
    echo "✅ Validation annotations found"
else
    echo "❌ Validation annotations not found at: data/bdd100k/labels/det_20/det_val.json"
fi

if [ -d "data/bdd100k/images/100k/train" ]; then
    train_count=$(ls -1 data/bdd100k/images/100k/train/*.jpg 2>/dev/null | wc -l)
    if [ $train_count -gt 0 ]; then
        echo "✅ Training images found ($train_count images)"
    else
        echo "⚠️  Training images directory exists but is empty"
    fi
else
    echo "❌ Training images directory not found"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo ""
echo "Directory structure: ✅"
echo "Python 3.9+: ✅"
echo "Virtual environment: ✅"
echo "Dependencies: ✅"
echo ""

if [ -f "data/bdd100k/labels/det_20/det_train.json" ]; then
    echo "Dataset: ✅ Ready to go!"
    echo ""
    echo "🚀 Next Steps:"
    echo ""
    echo "1. Activate virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run data analysis:"
    echo "   python data_analysis/analysis.py"
    echo ""
    echo "3. View interactive dashboard:"
    echo "   streamlit run data_analysis/dashboard.py"
    echo ""
    echo "4. Train model (if GPU available):"
    echo "   python model/train.py --model m --epochs 50"
    echo ""
    echo "OR use Docker:"
    echo "   docker-compose up data-analysis"
    echo "   docker-compose up dashboard"
    echo ""
else
    echo "Dataset: ❌ Please download BDD100K dataset"
    echo ""
    echo "📥 Download Instructions:"
    echo "   1. Visit: https://bdd-data.berkeley.edu/"
    echo "   2. Register and download:"
    echo "      - 100K Images"
    echo "      - Detection 2020 Labels"
    echo "   3. Extract files to data/bdd100k/"
    echo "   4. Re-run this setup script"
    echo ""
fi

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
