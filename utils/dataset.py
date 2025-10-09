import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class DriverDistractionDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        split_path = os.path.join(root_dir, split_file)
        
        self.data = []
        self.labels = []
        self.classes = []
        
        class_to_idx = {}
        
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 3:
                    img_path = parts[1]
                    label = int(parts[2])
                    
                    class_name = img_path.split('/')[0]
                    if class_name not in class_to_idx:
                        class_to_idx[class_name] = len(self.classes)
                        self.classes.append(class_name)
                    
                    full_path = os.path.join(root_dir, img_path)
                    if os.path.exists(full_path):
                        self.data.append(full_path)
                        self.labels.append(label)
        
        self.classes.sort()
        
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