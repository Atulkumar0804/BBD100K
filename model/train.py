"""
YOLOv11 Training Pipeline for BDD100K Object Detection

Comprehensive training script with:
- DataLoader setup
- Model initialization
- Optimizer and scheduler configuration
- Training loop with validation
- Checkpoint saving
- TensorBoard logging
- Loss tracking
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
import yaml
import os
import time
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import argparse

from model import BDD100KYOLOv11


class BDD100KTrainer:
    """Trainer class for YOLOv11 on BDD100K dataset."""
    
    def __init__(
        self,
        model_size: str = 'm',
        data_config: str = 'configs/bdd100k.yaml',
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        workers: int = 8,
        project: str = 'runs-model/train',
        name: str = 'bdd100k_yolo11',
        pretrained: bool = True,
        resume: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_size: YOLOv11 model size (n, s, m, l, x)
            data_config: Path to data configuration YAML
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            device: Device to train on
            workers: Number of dataloader workers
            project: Project directory for runs
            name: Run name
            pretrained: Use pretrained weights
            resume: Path to checkpoint to resume from
        """
        self.model_size = model_size
        self.data_config = data_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.workers = workers
        self.pretrained = pretrained
        self.resume = resume
        
        # Setup directories
        self.project_dir = Path(project)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.project_dir / f"{name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.weights_dir = self.run_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.run_dir / 'logs'))
        
        # Initialize model
        self.model = None
        self.setup_model()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        print(f"\n{'='*80}")
        print(f"Training Configuration:")
        print(f"  Model: YOLOv11{model_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Image Size: {img_size}")
        print(f"  Device: {device}")
        print(f"  Workers: {workers}")
        print(f"  Output: {self.run_dir}")
        print(f"{'='*80}\n")
    
    def setup_model(self):
        """Initialize YOLOv11 model."""
        print("Setting up model...")
        
        if self.resume:
            print(f"Resuming from checkpoint: {self.resume}")
            self.model = YOLO(self.resume)
        elif self.pretrained:
            model_name = f'yolo11{self.model_size}.pt'
            print(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        else:
            model_name = f'yolo11{self.model_size}.yaml'
            print(f"Initializing model from config: {model_name}")
            self.model = YOLO(model_name)
        
        print("✅ Model loaded successfully")
    
    def train(self):
        """
        Run training loop.
        
        Note: YOLOv11 from ultralytics has built-in training,
        so we use their optimized trainer.
        """
        print("\n🚀 Starting Training...\n")
        
        # Training configuration
        train_args = {
            'data': self.data_config,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'device': self.device,
            'workers': self.workers,
            'project': str(self.project_dir),
            'name': self.run_dir.name,
            'exist_ok': True,
            
            # Optimization
            'optimizer': 'AdamW',  # AdamW optimizer
            'lr0': 0.001,          # Initial learning rate
            'lrf': 0.01,           # Final learning rate (lr0 * lrf)
            'momentum': 0.937,     # SGD momentum/Adam beta1
            'weight_decay': 0.0005,  # Optimizer weight decay
            
            # Augmentation
            'hsv_h': 0.015,        # HSV-Hue augmentation
            'hsv_s': 0.7,          # HSV-Saturation augmentation
            'hsv_v': 0.4,          # HSV-Value augmentation
            'degrees': 0.0,        # Rotation (+/- deg)
            'translate': 0.1,      # Translation (+/- fraction)
            'scale': 0.5,          # Scaling (+/- gain)
            'shear': 0.0,          # Shear (+/- deg)
            'perspective': 0.0,    # Perspective (+/- fraction)
            'flipud': 0.0,         # Flip up-down (probability)
            'fliplr': 0.5,         # Flip left-right (probability)
            'mosaic': 1.0,         # Mosaic augmentation (probability)
            'mixup': 0.0,          # MixUp augmentation (probability)
            'copy_paste': 0.0,     # Copy-paste augmentation (probability)
            
            # Training settings
            'patience': 50,        # Early stopping patience
            'save': True,          # Save checkpoints
            'save_period': -1,     # Disable periodic checkpoint saving to save space
            'cache': False,        # Cache images for faster training
            'amp': True,           # Automatic Mixed Precision
            'fraction': 1.0,       # Dataset fraction to train on
            'profile': False,      # Profile ONNX and TensorRT speeds
            'freeze': None,        # Freeze layers (None or list of layer indices)
            
            # Validation
            'val': True,           # Validate during training
            'plots': True,         # Save plots
            'verbose': True,       # Verbose output
        }
        
        # If resuming, add resume=True to args
        if self.resume:
            train_args['resume'] = True
        
        # Start training
        results = self.model.train(**train_args)
        
        print("\n✅ Training completed!")
        
        # Save final model
        final_model_path = self.weights_dir / 'best.pt'
        print(f"Best model saved to: {final_model_path}")
        
        # Create a copy to a generic location for easier access
        generic_best_path = self.project_dir / 'best.pt'
        if final_model_path.exists():
            try:
                import shutil
                shutil.copy2(final_model_path, generic_best_path)
                print(f"Copy of best model saved to: {generic_best_path}")
            except Exception as e:
                print(f"Could not create generic copy of best model: {e}")
        
        return results
    
    def evaluate(self, val_data=None):
        """
        Evaluate model on validation set.
        
        Args:
            val_data: Path to validation data (uses config if None)
        """
        print("\n📊 Evaluating model...")
        
        val_args = {
            'data': val_data or self.data_config,
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'device': self.device,
            'plots': True,
            'save_json': True,
            'verbose': True
        }
        
        metrics = self.model.val(**val_args)
        
        print("\n✅ Evaluation completed!")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        return metrics
    
    def save_training_summary(self):
        """Save training summary and configuration."""
        summary = {
            'model_size': self.model_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'device': self.device,
            'pretrained': self.pretrained,
            'run_dir': str(self.run_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.run_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to: {summary_path}")


def create_data_config(
    train_images: str,
    val_images: str,
    train_labels: str,
    val_labels: str,
    output_path: str = 'configs/bdd100k.yaml'
):
    """
    Create YOLOv11 data configuration file.
    
    Args:
        train_images: Path to training images
        val_images: Path to validation images
        train_labels: Path to training labels
        val_labels: Path to validation labels
        output_path: Where to save config
    """
    config = {
        'path': str(Path(train_images).parent.parent),
        'train': str(Path(train_images).relative_to(Path(train_images).parent.parent)),
        'val': str(Path(val_images).relative_to(Path(val_images).parent.parent)),
        
        'nc': 10,
        'names': [
            'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motor', 'bike', 'traffic light', 'traffic sign'
        ]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Data config saved to: {output_path}")
    return output_path


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train YOLOv11 on BDD100K')
    
    parser.add_argument('--model', type=str, default='m', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size')
    parser.add_argument('--data', type=str, default='configs/bdd100k.yaml',
                       help='Path to data config')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of workers')
    parser.add_argument('--project', type=str, default=None,
                       help='Project directory (default: <project_root>/runs-model/train)')
    parser.add_argument('--name', type=str, default='bdd100k_yolo11',
                       help='Run name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate, no training')
    
    args = parser.parse_args()

    # Resolve project directory relative to project root if not specified
    if args.project is None:
        args.project = str(Path(__file__).resolve().parent.parent / "runs" / "train")
    
    # Check if data config exists
    if not os.path.exists(args.data):
        print(f"⚠️  Data config not found: {args.data}")
        print("Creating default config...")
        
        # You need to update these paths
        train_images = 'data/bdd100k/images/100k/train'
        val_images = 'data/bdd100k/images/100k/val'
        
        create_data_config(
            train_images=train_images,
            val_images=val_images,
            train_labels='',  # YOLOv11 expects labels in images/../labels/
            val_labels='',
            output_path=args.data
        )
    
    # Initialize trainer
    trainer = BDD100KTrainer(
        model_size=args.model,
        data_config=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        resume=args.resume
    )
    
    if args.eval_only:
        # Evaluation only
        trainer.evaluate()
    else:
        # Train
        results = trainer.train()
        
        # Evaluate
        trainer.evaluate()
        
        # Save summary
        trainer.save_training_summary()
    
    print("\n🎉 All done!\n")


if __name__ == "__main__":
    main()
