import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class DriverDistractionDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, use_percent=100):
        self.root_dir = root_dir
        self.transform = transform

        split_path = os.path.join(root_dir, split_file)

        self.data = []
        self.labels = []

        # Build mapping from label indices to class names
        label_to_class = {}

        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) >= 3:
                    img_path = parts[1]
                    label = int(parts[2])

                    # Extract class name from image path
                    class_name = img_path.split('/')[0]

                    # Map label index to class name (preserves original ordering)
                    if label not in label_to_class:
                        label_to_class[label] = class_name

                    full_path = os.path.join(root_dir, img_path)
                    if os.path.exists(full_path):
                        self.data.append(full_path)
                        self.labels.append(label)

        # Apply percentage sampling if use_percent < 100
        if use_percent < 100:
            total_samples = len(self.data)
            num_samples = int(total_samples * use_percent / 100)
            # Use first num_samples to maintain deterministic behavior
            self.data = self.data[:num_samples]
            self.labels = self.labels[:num_samples]

        # Create ordered list of class names based on label indices
        max_label = max(label_to_class.keys()) if label_to_class else -1
        self.classes = [label_to_class.get(i, f"class_{i}") for i in range(max_label + 1)]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label