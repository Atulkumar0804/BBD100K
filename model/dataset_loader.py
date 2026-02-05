import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json

class BDD100KDataset(Dataset):
    """
    PyTorch Dataset for BDD100K object detection to demonstrate data loading capabilities.
    """
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            annotation_file (str): Path to the JSON file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Filter out images that don't exist
        self.valid_annotations = [
            ann for ann in self.annotations 
            if os.path.exists(os.path.join(self.image_dir, ann['name']))
        ]

    def __len__(self):
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        annotation = self.valid_annotations[idx]
        img_name = annotation['name']
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        
        # Load Targets (Bounding Boxes and Labels)
        boxes = []
        labels = []
        
        if 'labels' in annotation:
            for obj in annotation['labels']:
                if 'box2d' in obj:
                    # BDD format is [x1, y1, x2, y2]
                    b = obj['box2d']
                    boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
                    labels.append(obj['category'])

        # Convert to tensors
        start_tensors_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Note: In a real training pipeline, labels would need to be integer encoded
        # specific_labels = encoder(labels) 
        
        target = {}
        target["boxes"] = start_tensors_boxes
        target["labels"] = labels # Returning raw strings for demonstration
        target["image_id"] = torch.tensor([idx])
        
        if self.transform:
            image = self.transform(image)

        return image, target

# Example Usage Block (Not executed on import)
if __name__ == "__main__":
    # Hypothetical paths
    dataset = BDD100KDataset(
        image_dir="data/bdd100k/images/100k/train",
        annotation_file="data/bdd100k/labels/bdd100k_labels_images_train.json"
    )
    print(f"Dataset size: {len(dataset)}")
