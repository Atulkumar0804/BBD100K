# ============================================================================
# BDD100K Object Detection - Complete All-in-One Dockerfile
# Includes: Data Analysis, Model Training, Inference, Evaluation, Dashboard
# Base: PyTorch 2.1.0 with CUDA 11.8
# ============================================================================

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="BDD100K Project"
LABEL description="Complete BDD100K Object Detection Pipeline - All Components"

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

# Set timezone
RUN echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y tzdata && \
    rm /etc/timezone && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean

# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    vim \
    nano \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================================================
# WORKING DIRECTORY
# ============================================================================
WORKDIR /app

# ============================================================================
# PYTHON DEPENDENCIES - All Components
# ============================================================================

# Upgrade pip and setuptools (keep wheel as provided by base image)
RUN pip install --no-cache-dir --upgrade pip setuptools

# ML/DL Core Frameworks
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    ultralytics>=8.3.0

# Data Processing & Numerical Computing
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    opencv-python==4.8.0.76 \
    Pillow==9.5.0 \
    scipy==1.11.2 \
    scikit-learn==1.3.0 \
    albumentations==1.3.1

# Visualization & Dashboard
RUN pip install --no-cache-dir \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.16.1 \
    streamlit==1.25.0

# Training & Monitoring
RUN pip install --no-cache-dir \
    tensorboard==2.14.0 \
    tqdm==4.66.1 \
    pyyaml==6.0.1

# Development & Notebooks
RUN pip install --no-cache-dir \
    jupyter==1.0.0 \
    jupyterlab==4.0.6 \
    ipython==8.15.0 \
    black==23.9.1 \
    pylint==3.0.1

# ============================================================================
# COPY PROJECT FILES AND DIRECTORIES
# ============================================================================

# Configuration
COPY configs/ /app/configs/

# Source code directories
COPY data_analysis/ /app/data_analysis/
COPY model/ /app/model/
COPY evaluation/ /app/evaluation/
COPY notebooks/ /app/notebooks/

# Scripts and documentation
COPY *.sh /app/
COPY *.txt /app/
COPY *.md /app/
COPY *.yml /app/
COPY docker-compose.yml /app/

# Requirements files for reference
COPY requirements*.txt /app/

# ============================================================================
# CREATE NECESSARY DIRECTORIES AND PERMISSIONS
# ============================================================================
RUN mkdir -p /app/data/bdd100k/images/100k/train /app/data/bdd100k/images/100k/val /app/data/bdd100k/images/100k/test && \
    mkdir -p /app/data/bdd100k/labels/100k/train /app/data/bdd100k/labels/100k/val && \
    mkdir -p /app/runs-model/train && \
    mkdir -p /app/runs-model/detect && \
    mkdir -p /app/output-Data_Analysis && \
    chmod +x /app/*.sh || true

# ============================================================================
# COPY ENTRYPOINT SCRIPT
# ============================================================================
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# ============================================================================
# ENTRYPOINT AND DEFAULT COMMAND
# ============================================================================
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["help"]
