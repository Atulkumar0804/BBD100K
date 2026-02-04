"""
Convert BDD100K JSON labels to YOLO format

BDD100K format:
- JSON file with image annotations
- Bounding boxes in absolute pixel coordinates

YOLO format:
- One .txt file per image
- Each line: <class_id> <x_center> <y_center> <width> <height>
- All coordinates normalized to [0, 1]
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

# BDD100K class mapping (All 10 object detection classes)
# Excluding: drivable area, lane (segmentation only)
CLASS_MAPPING = {
    'person': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motor': 6,
    'bike': 7,
    'traffic light': 8,
    'traffic sign': 9
}


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert BDD100K bbox to YOLO format.
    
    BDD100K: {x1, y1, x2, y2} (top-left, bottom-right)
    YOLO: <x_center> <y_center> <width> <height> (all normalized)
    """
    x1 = bbox['x1']
    y1 = bbox['y1']
    x2 = bbox['x2']
    y2 = bbox['y2']
    
    # Calculate center, width, height
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clip to [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def convert_bdd100k_to_yolo(json_path, output_dir, img_width=1280, img_height=720):
    """
    Convert BDD100K JSON annotations to YOLO format txt files.
    
    Args:
        json_path: Path to BDD100K JSON file
        output_dir: Directory to save YOLO label files
        img_width: Image width (default: 1280 for BDD100K)
        img_height: Image height (default: 720 for BDD100K)
    """
    # Load JSON
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    total_images = len(data)
    total_objects = 0
    images_with_labels = 0
    
    print(f"Converting {total_images} images to YOLO format...")
    
    for image_data in tqdm(data, desc="Processing"):
        image_name = image_data['name']
        base_name = Path(image_name).stem  # Remove extension
        
        # YOLO label file path
        label_file = output_dir / f"{base_name}.txt"
        
        # Get labels
        labels = image_data.get('labels', [])
        
        if not labels:
            # Create empty file for images without labels
            label_file.touch()
            continue
        
        # Convert each label
        yolo_lines = []
        for label in labels:
            category = label.get('category', '')
            
            # Skip if category not in our mapping
            if category not in CLASS_MAPPING:
                continue
            
            # Get bbox
            box2d = label.get('box2d', None)
            if box2d is None:
                continue
            
            class_id = CLASS_MAPPING[category]
            
            try:
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    box2d, img_width, img_height
                )
                
                # Write in YOLO format
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                yolo_lines.append(yolo_line)
                total_objects += 1
                
            except Exception as e:
                print(f"Error converting bbox in {image_name}: {e}")
                continue
        
        # Write label file
        if yolo_lines:
            with open(label_file, 'w') as f:
                f.writelines(yolo_lines)
            images_with_labels += 1
        else:
            # Create empty file
            label_file.touch()
    
    print("\nConversion complete!")
    print(f"   Total images: {total_images}")
    print(f"   Images with labels: {images_with_labels}")
    print(f"   Total objects: {total_objects}")
    print(f"   Labels saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert BDD100K to YOLO format")
    parser.add_argument('--train-json', type=str,
                        default='data/bdd100k/labels/bdd100k_labels_images_train.json',
                        help='Path to training JSON')
    parser.add_argument('--val-json', type=str,
                        default='data/bdd100k/labels/bdd100k_labels_images_val.json',
                        help='Path to validation JSON')
    parser.add_argument('--train-output', type=str,
                        default='data/bdd100k/labels/100k/train',
                        help='Output directory for training labels')
    parser.add_argument('--val-output', type=str,
                        default='data/bdd100k/labels/100k/val',
                        help='Output directory for validation labels')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BDD100K to YOLO Format Converter")
    print("="*80)
    
    # Convert training set
    print("\nConverting TRAINING set...")
    convert_bdd100k_to_yolo(
        json_path=args.train_json,
        output_dir=args.train_output
    )
    
    # Convert validation set
    print("\nConverting VALIDATION set...")
    convert_bdd100k_to_yolo(
        json_path=args.val_json,
        output_dir=args.val_output
    )
    
    print("\n" + "="*80)
    print("All conversions complete! Ready for YOLO training.")
    print("="*80)
