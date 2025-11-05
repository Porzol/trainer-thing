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
        label_to_class = {}
        skipped_files = []

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
                    if label not in label_to_class:
                        label_to_class[label] = class_name
                    full_path = os.path.join(root_dir, img_path)

                    # Skip corrupted files (e.g., .tmp files)
                    if img_path.endswith('.tmp'):
                        skipped_files.append((full_path, 'Skipped .tmp file'))
                        continue

                    if os.path.exists(full_path):
                        # Verify the image can be loaded before adding to dataset
                        try:
                            with Image.open(full_path) as img:
                                img.verify()
                            # Add to dataset only if image is valid
                            self.data.append(full_path)
                            self.labels.append(label)
                        except Exception as e:
                            skipped_files.append((full_path, str(e)))

        # Log skipped files
        if skipped_files:
            print(f"\nWarning: Skipped {len(skipped_files)} corrupted/invalid files from {split_file}:")
            for path, reason in skipped_files[:10]:  # Show first 10
                print(f"  - {path}: {reason}")
            if len(skipped_files) > 10:
                print(f"  ... and {len(skipped_files) - 10} more")

        if use_percent < 100:
            num_samples = int(len(self.data) * use_percent / 100)
            self.data = self.data[:num_samples]
            self.labels = self.labels[:num_samples]

        max_label = max(label_to_class.keys()) if label_to_class else -1
        self.classes = [label_to_class.get(i, f"class_{i}") for i in range(max_label + 1)]

        print(f"Loaded {len(self.data)} valid samples from {split_file}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # All images should be valid since we verified them in __init__
        # If an image fails here, it's a serious error that should raise an exception
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label