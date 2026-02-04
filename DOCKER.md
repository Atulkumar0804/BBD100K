
# 🐳 Docker Setup Guide - BDD100K Object Detection

## Complete All-in-One Docker Solution

This guide covers the complete, production-ready Docker setup for the BDD100K Object Detection project.

---

## 📋 What's Included

### Single Unified Dockerfile
- ✅ All project files: `data_analysis/`, `model/`, `evaluation/`, `notebooks/`, `configs/`
- ✅ All Python dependencies pre-installed
- ✅ CUDA 11.8 + PyTorch 2.1.0 + cuDNN 8
- ✅ Smart entrypoint script with 10+ commands
- ✅ Support for all services: analysis, training, inference, evaluation, dashboard, jupyter, tensorboard
- ✅ GPU support (NVIDIA)
- ✅ ~6-8GB image size

### Docker Compose Configuration
- ✅ 8 pre-configured services
- ✅ Automatic service dependencies
- ✅ Network isolation with custom bridge network
- ✅ Volume management for data persistence
- ✅ GPU allocation for training/inference services

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Build Docker Image
```bash
cd /home/atul/Desktop/atul/Bosch/bosch-bdd-object-detection

# Build the unified image (first time only, takes 15-20 minutes)
docker build -t bdd100k:latest -f Dockerfile .
```

### Step 2: Verify Build
```bash
# Check image exists
docker images | grep bdd100k

# Test image works
docker run --rm bdd100k:latest help
```

### Step 3: Run Services

**Option A: Use docker-compose (All services)**
```bash
docker-compose up -d
```

**Option B: Run individual services**
```bash
# Data Analysis
docker run -it --rm \
  -v ./data:/app/data:ro \
  -v ./output-Data_Analysis:/app/output-Data_Analysis:rw \
  bdd100k:latest analysis

# Dashboard
docker run -it --rm -p 8501:8501 \
  -v ./output-Data_Analysis:/app/output-Data_Analysis:ro \
  bdd100k:latest dashboard
```

### Step 4: Access Results
- **Dashboard**: http://localhost:8501
- **Jupyter**: http://localhost:8888
- **TensorBoard**: http://localhost:6006

---

## 📝 Available Docker Commands

### View All Commands
```bash
docker run --rm bdd100k:latest help
```

### Data & Analysis
```bash
# Run complete data analysis pipeline
docker run -it --rm \
  -v ./data:/app/data:ro \
  -v ./output-Data_Analysis:/app/output-Data_Analysis:rw \
  bdd100k:latest analysis
```

### Training

**YOLO11 Training (GPU Required)**
```bash
docker run -it --rm --gpus all \
  -v ./data:/app/data:ro \
  -v ./runs:/app/runs:rw \
  bdd100k:latest train
```

### Inference & Evaluation

**Run Inference**
```bash
docker run -it --rm --gpus all \
  -v ./data:/app/data:ro \
  -v ./runs:/app/runs:ro \
  bdd100k:latest inference
```

**Evaluate Model**
```bash
docker run -it --rm \
  -v ./runs:/app/runs:ro \
  -v ./output-Data_Analysis:/app/output-Data_Analysis:rw \
  bdd100k:latest evaluate
```

**Complete Pipeline**
```bash
docker run -it --rm --gpus all \
  -v ./data:/app/data:ro \
  -v ./runs:/app/runs:rw \
  -v ./outputs:/app/outputs:rw \
  -v ./output-Data_Analysis:/app/output-Data_Analysis:rw \
  bdd100k:latest pipeline
```

### Interactive Services

**Dashboard (Streamlit)**
```bash
docker run -it --rm -p 8501:8501 \
  -v ./output-Data_Analysis:/app/output-Data_Analysis:ro \
  bdd100k:latest dashboard
# Open: http://localhost:8501
```

**Jupyter Notebook**
```bash
docker run -it --rm -p 8888:8888 \
  -v ./notebooks:/app/notebooks:rw \
  bdd100k:latest jupyter
# Open: http://localhost:8888
```

**TensorBoard**
```bash
docker run -it --rm -p 6006:6006 \
  -v ./runs:/app/runs:ro \
  bdd100k:latest tensorboard
# Open: http://localhost:6006
```

### System Commands

**Bash Shell**
```bash
docker run -it --rm \
  -v $(pwd):/app \
  bdd100k:latest bash
```

**Python Interpreter**
```bash
docker run -it --rm bdd100k:latest python
```

---

## 🐳 Docker Compose Services

### Service Overview

```bash
# Start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Individual Services

| Service | Command | Purpose | GPU | Port |
|---------|---------|---------|-----|------|
| analysis | `docker-compose up analysis` | Data analysis | ❌ | - |
| train-yolo | `docker-compose up train-yolo` | YOLO11 training | ✅ | - |
| inference | `docker-compose up inference` | Run predictions | ✅ | - |
| evaluate | `docker-compose up evaluate` | Calculate metrics | ❌ | - |
| dashboard | `docker-compose up dashboard` | Streamlit UI | ❌ | 8501 |
| jupyter | `docker-compose up jupyter` | Notebooks | ❌ | 8888 |
| tensorboard | `docker-compose up tensorboard` | Training monitor | ❌ | 6006 |

### Service Dependencies

```
analysis
  ↓
  └─→ train-yolo ─→ inference ─→ evaluate
  │
  └─→ dashboard, jupyter, tensorboard
```

### Compose Management

```bash
# View logs for specific service
docker-compose logs -f dashboard
docker-compose logs -f train-yolo

# Restart service
docker-compose restart dashboard

# Scale services (parallel processing)
docker-compose up -d --scale train-yolo=2

# Stop specific service
docker-compose stop train-yolo

# Remove everything (containers, volumes)
docker-compose down -v
```

---

## 💾 Volume Management

### Mounted Volumes

| Container Path | Host Path | Mode | Purpose |
|---|---|---|---|
| `/app/data` | `./data` | ro | Input dataset (read-only) |
| `/app/runs` | `./runs` | rw | YOLO training outputs |
| `/app/outputs` | `./outputs` | rw | Torchvision outputs |
| `/app/output-Data_Analysis` | `./output-Data_Analysis` | rw | Analysis results |
| `/app/notebooks` | `./notebooks` | rw | Jupyter notebooks |
| `/app/configs` | `./configs` | ro | Configuration files |

### Volume Best Practices

```bash
# Read-only data volumes (safer)
-v ./data:/app/data:ro

# Read-write output volumes (results saved)
-v ./output-Data_Analysis:/app/output-Data_Analysis:rw

# Full project directory (development)
-v $(pwd):/app:rw
```

---

## 🔧 Advanced Configuration

### Environment Variables

```bash
# GPU Selection (0=first GPU, 1=second GPU, etc.)
-e CUDA_VISIBLE_DEVICES=0

# Python Optimization
-e PYTHONUNBUFFERED=1
-e PYTHONDONTWRITEBYTECODE=1

# Custom environment
docker run -e MY_VAR=value bdd100k:latest bash
```

### Resource Limits

```bash
docker run \
  --gpus all \
  --cpus=4 \
  --memory=16g \
  -it bdd100k:latest bash
```

### Port Mapping

```bash
# Different ports
docker run -p 8502:8501 bdd100k:latest dashboard  # Maps 8502→8501
docker run -p 8889:8888 bdd100k:latest jupyter    # Maps 8889→8888
docker run -p 6007:6006 bdd100k:latest tensorboard # Maps 6007→6006
```

---

## 🔍 Monitoring & Debugging

### View Logs

```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail 100

# Specific service
docker-compose logs -f train-yolo

# Timestamps
docker-compose logs --timestamps

# View logs from stopped container
docker logs <container-id>
```

### Inspect Container

```bash
# Enter running container
docker exec -it bdd100k-analysis bash

# View container details
docker inspect bdd100k-analysis

# View resource usage
docker stats

# View container processes
docker top <container-id>
```

### Check Image Details

```bash
# View image layers
docker history bdd100k:latest

# View image metadata
docker inspect bdd100k:latest

# View image size
docker images bdd100k:latest
```

---

## 🐛 Troubleshooting

### Issue: GPU Not Detected

**Symptoms:** CUDA not available inside container

**Solution:**
```bash
# Check NVIDIA Docker installation
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi

# If above works but your container doesn't see GPU:
# Check Dockerfile uses correct base image:
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime ✓

# Use correct run flag:
docker run --gpus all bdd100k:latest python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Out of Memory

**Symptoms:** Process killed, OOM errors

**Solution:**
```bash
# Reduce batch size
docker run --gpus all bdd100k:latest bash
python model/train.py --batch 8  # Reduce from 16

# Or limit memory
docker run --gpus all --memory=16g bdd100k:latest train-yolo

# Or increase swap
docker run --gpus all --memory-swap=-1 bdd100k:latest train-yolo
```

### Issue: Port Already in Use

**Symptoms:** `Address already in use` error

**Solution:**
```bash
# Use different port
docker run -p 8502:8501 bdd100k:latest dashboard

# Or find and stop conflicting container
lsof -i :8501
docker kill <container-id>
```

### Issue: Slow Data Loading

**Symptoms:** Training/inference is slow

**Solution:**
```bash
# Use local volumes, not network drives
# ❌ Avoid: -v //wsl$/Ubuntu/path:/app
# ✅ Use: -v /local/path:/app

# Or copy data into container
docker cp ./data bdd100k-container:/app/data
```

### Issue: Permission Denied

**Symptoms:** `Permission denied` errors

**Solution:**
```bash
# Fix file permissions on host
chmod -R 755 ./data ./runs ./outputs

# Or use Docker without host mounts
docker run --rm bdd100k:latest bash -c "cd /app && python model/train.py"
```

### Issue: Build Fails

**Symptoms:** Docker build errors

**Solution:**
```bash
# Retry with verbose output
docker build --progress=plain -t bdd100k:latest -f Dockerfile .

# Check internet connection
ping google.com

# Clear Docker cache
docker system prune -a

# Check available disk space
df -h
```

---

## 📊 Docker Compose Workflows

### Workflow 1: Complete Pipeline (Sequential)

```bash
# Run all services in order
docker-compose up analysis train-yolo inference evaluate

# Monitor progress
docker-compose logs -f
```

### Workflow 2: Development Mode

```bash
# Start interactive bash with all mounts
docker run -it --rm \
  -v $(pwd):/app \
  --gpus all \
  bdd100k:latest bash

# Inside container: edit and run commands
python model/train.py --epochs 1 --batch 8
```

### Workflow 3: Dashboard + Training (Parallel)

```bash
# Terminal 1: Start dashboard
docker run -it --rm -p 8501:8501 \
  -v ./output-Data_Analysis:/app/output-Data_Analysis:ro \
  bdd100k:latest dashboard

# Terminal 2: Start training
docker run -it --rm --gpus all \
  -v ./data:/app/data:ro \
  -v ./runs:/app/runs:rw \
  bdd100k:latest train-yolo

# Terminal 3: Monitor with TensorBoard
docker run -it --rm -p 6006:6006 \
  -v ./runs:/app/runs:ro \
  bdd100k:latest tensorboard
```

### Workflow 4: Batch Inference

```bash
# Process many images in sequence
docker run -it --rm --gpus all \
  -v ./data:/app/data:ro \
  -v ./outputs:/app/outputs:rw \
  bdd100k:latest bash

# Inside container
python model/inference.py --model /app/runs/train/best.pt --source /app/data/images
```

---

## 🧹 Cleanup

### Remove Containers

```bash
# Stop and remove specific container
docker-compose down

# Remove all project containers
docker rm bdd100k-*

# Force remove running containers
docker rm -f bdd100k-*
```

### Remove Images

```bash
# Remove Docker image
docker rmi bdd100k:latest

# Force remove
docker rmi -f bdd100k:latest

# Remove dangling images
docker image prune -a
```

### System Cleanup

```bash
# Remove unused containers, images, volumes
docker system prune -a

# Remove unused volumes
docker volume prune

# Check disk usage
docker system df
```

---

## 📈 Performance Optimization

### Build Optimization

```bash
# Multi-stage build (future improvement)
# FROM pytorch/pytorch as base
# ...
# FROM base as final

# Use layer caching effectively
# Place frequently-changing commands at end

# Current build time: ~15-20 minutes
```

### Runtime Optimization

```bash
# Use CPU optimization
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Set number of workers
export OMP_NUM_THREADS=8

# Optimize disk I/O
docker run --io-maxbandwidth=1gb bdd100k:latest train-yolo
```

### Memory Optimization

```bash
# Check memory usage
docker stats

# Limit memory to prevent OOM
docker run --memory=16g bdd100k:latest train-yolo

# Enable memory swap
docker run --memory=16g --memory-swap=32g bdd100k:latest train-yolo
```

---

## 🔐 Security Best Practices

### Don't Run as Root

```dockerfile
# Future improvement: Add non-root user
# RUN useradd -m -s /bin/bash appuser
# USER appuser
```

### Read-Only Volumes

```bash
# Use read-only mounts for data
-v ./data:/app/data:ro
```

### Network Isolation

```bash
# Custom network (already in compose)
networks:
  bdd100k-network:
    driver: bridge
```

---

## 📚 Reference

### Docker Commands Cheat Sheet

```bash
# Build
docker build -t name:tag -f Dockerfile .

# Run
docker run [OPTIONS] image command

# Common options
--rm                    # Remove container after exit
-it                    # Interactive + TTY
-d                     # Detached (background)
-p port:port           # Port mapping
-v path:path           # Volume mount
-e VAR=value           # Environment variable
--gpus all             # Enable GPU
--name container-name  # Container name

# Docker Compose
docker-compose build            # Build images
docker-compose up               # Start services
docker-compose up -d            # Start in background
docker-compose down             # Stop services
docker-compose logs -f          # View logs
docker-compose exec svc cmd     # Execute in service
docker-compose restart svc      # Restart service
```

### Useful Flags

```bash
# Limit resources
--cpus=4               # Limit CPU cores
--memory=16g           # Limit memory

# Networking
--network host         # Use host network (faster)
--network-alias name   # DNS alias

# Volumes
-v host:container:ro   # Read-only
-v host:container:rw   # Read-write (default)
-v name:/path          # Named volume
```

---

## ✅ Verification Checklist

Before using Docker, verify:

- [ ] Docker installed: `docker --version`
- [ ] Docker running: `docker ps`
- [ ] Docker Compose installed: `docker compose version`
- [ ] NVIDIA Docker (for GPU): `nvidia-docker --version`
- [ ] Sufficient disk space: `df -h` (50GB+)
- [ ] Sufficient RAM: `free -h` (16GB+)
- [ ] Internet connection (for first build)

After building image:

- [ ] Image exists: `docker images | grep bdd100k`
- [ ] Image works: `docker run --rm bdd100k:latest help`
- [ ] GPU detected: `docker run --rm --gpus all bdd100k:latest python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Data accessible: `docker run --rm -v ./data:/app/data:ro bdd100k:latest ls /app/data`

---

## 🎯 Next Steps

1. **Build Image**: `docker build -t bdd100k:latest -f Dockerfile .`
2. **Run Analysis**: `docker run -it --rm bdd100k:latest analysis`
3. **View Dashboard**: `docker run -it -p 8501:8501 bdd100k:latest dashboard`
4. **Train Model**: `docker run -it --gpus all bdd100k:latest train-yolo`
5. **Use docker-compose**: `docker-compose up -d`

---

## 📞 Support

For issues:
1. Check logs: `docker-compose logs -f [service]`
2. See troubleshooting section above
3. Check Docker/Compose versions match requirements
4. Verify sufficient disk space and RAM
5. Ensure NVIDIA Docker works: `nvidia-docker run --rm nvidia/cuda:11.8.0-runtime nvidia-smi`

---

**Happy containerizing! 🚀**
