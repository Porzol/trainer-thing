import os
import json
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import onnx
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

from models.efficientnet import EfficientNetModel
from models.mobilenetv3 import MobileNetV3Model
from models.regnet import RegNetModel
from models.resnet import ResNetV2Model
from utils.dataset import DriverDistractionDataset
from utils.metrics import MetricsTracker
from utils.visualizations import create_confusion_matrix, create_per_class_metrics, save_sample_predictions

def get_optimal_num_workers():
    """Automatically determine optimal number of workers for DataLoader.

    Returns safe defaults that balance performance and stability:
    - Respects DATALOADER_NUM_WORKERS env var if set
    - Uses 75% of available CPUs by default
    """
    # Allow explicit override via environment variable
    if 'DATALOADER_NUM_WORKERS' in os.environ:
        try:
            return int(os.environ['DATALOADER_NUM_WORKERS'])
        except ValueError:
            pass

    # Get CPU count
    cpu_count = os.cpu_count() or 4

    # Use 75% of available CPUs
    num_workers = max(1, int(cpu_count * 0.75))

    return num_workers

class DistractionTrainer:
    def __init__(self, config_path, dataset_path=None, force_cpu=False):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.dataset_path = dataset_path or self.config.get('dataset_path', 'Dataset/Cam2')
        self.device = torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 22
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f"results/{self.config['run_name']}_{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_f1 = 0.0
        self.patience = self.config.get('patience', 0)
        self.patience_counter = 0
        self.metrics_tracker = MetricsTracker()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        
    def _setup_model(self):
        model_name = self.config['model']
        dropout = self.config.get('dropout', 0.5)
        activation = self.config.get('activation_function', 'relu')
        
        if model_name == 'efficientnet':
            self.model = EfficientNetModel(self.num_classes, dropout, activation)
        elif model_name == 'mobilenetv3':
            self.model = MobileNetV3Model(self.num_classes, dropout, activation)
        elif model_name == 'regnet':
            self.model = RegNetModel(self.num_classes, dropout, activation)
        elif model_name == 'resnetv2':
            self.model = ResNetV2Model(self.num_classes, dropout, activation)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        
    def _setup_data(self):
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        training_use_percent = self.config.get('training_use_percent', 100)
        self.train_dataset = DriverDistractionDataset(self.dataset_path, 'driver_train.txt', transform_train, training_use_percent)
        self.val_dataset = DriverDistractionDataset(self.dataset_path, 'driver_val.txt', transform_val)
        self.test_dataset = DriverDistractionDataset(self.dataset_path, 'driver_test.txt', transform_val)
        batch_size = self.config['batch_size']

        # Determine optimal num_workers: config > auto-detection
        num_workers = self.config.get('num_workers', get_optimal_num_workers())

        # Additional DataLoader settings for stability and performance
        pin_memory = self.device.type == 'cuda'
        persistent_workers = num_workers > 0  # Keep workers alive between epochs
        prefetch_factor = 2 if num_workers > 0 else None  # Prefetch batches per worker

        print(f"DataLoader configuration: num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            prefetch_factor=prefetch_factor
        )
        
    def _setup_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        start_time = time.time()
        pbar = tqdm(self.train_loader, desc="Training", leave=True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        epoch_time = time.time() - start_time
        return {
            'loss': running_loss / len(self.train_loader),
            'accuracy': accuracy_score(all_targets, all_preds),
            'f1_score': f1_score(all_targets, all_preds, average='weighted'),
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'training_time': epoch_time
        }
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=True)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return {
            'loss': running_loss / len(self.val_loader),
            'accuracy': accuracy_score(all_targets, all_preds),
            'f1_score': f1_score(all_targets, all_preds, average='weighted'),
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def test_model(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        return {
            'accuracy': accuracy_score(all_targets, all_preds),
            'f1_score': f1_score(all_targets, all_preds, average='weighted'),
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint_dir = self.run_dir / ("best" if is_best else f"epoch_{epoch:02d}")
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, checkpoint_dir / 'model.pth')
        self._export_onnx(checkpoint_dir / 'model.onnx')

        if is_best:
            with open(checkpoint_dir / 'epoch_info.txt', 'w') as f:
                f.write(f"Best model from epoch {epoch}\n")
                f.write(f"Val F1: {metrics['val']['f1_score']:.4f}\n")
                f.write(f"Val Acc: {metrics['val']['accuracy']:.4f}\n")
                f.write(f"Val Loss: {metrics['val']['loss']:.4f}\n")
        
    def _export_onnx(self, onnx_path):
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.model.eval()
        torch.onnx.export(self.model, dummy_input, onnx_path, export_params=True, opset_version=11,
                         do_constant_folding=True, input_names=['input'], output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
    def save_visualizations(self, epoch, val_metrics, is_best=False):
        vis_dir = self.run_dir / ("best" if is_best else f"epoch_{epoch:02d}") / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print("  - Creating confusion matrix...")
        create_confusion_matrix(val_metrics['targets'], val_metrics['predictions'],
                               self.train_dataset.classes, vis_dir / 'confusion_matrix.png')
        print("  - Creating per-class metrics...")
        create_per_class_metrics(val_metrics['targets'], val_metrics['predictions'],
                                 self.train_dataset.classes, vis_dir / 'per_class_metrics.png')
        print("  - Saving metrics history...")
        self.metrics_tracker.save_metrics_plot(vis_dir / 'metrics_history.png')
        print("  - Saving sample predictions...")
        save_sample_predictions(self.model, self.test_loader, self.device,
                               self.train_dataset.classes, vis_dir / 'sample_predictions.png')
        print("  - All visualizations saved.")
        
    def train(self):
        print(f"Starting training with config: {self.config}")
        print(f"Results will be saved to: {self.run_dir}")
        print(f"Using device: {self.device}")
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            self.metrics_tracker.update(epoch + 1, train_metrics, val_metrics)
            self.scheduler.step(val_metrics['f1_score'])
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            is_best = val_metrics['f1_score'] > self.best_f1
            if is_best:
                self.best_f1 = val_metrics['f1_score']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            should_checkpoint = (epoch + 1) % self.config['checkpoint_interval'] == 0
            metrics_dict = {'train': train_metrics, 'val': val_metrics}

            if should_checkpoint:
                print(f"Saving checkpoint for epoch {epoch + 1}...")
                self.save_checkpoint(epoch + 1, metrics_dict, is_best=False)
                print(f"Checkpoint saved. Generating visualizations...")
                self.save_visualizations(epoch + 1, val_metrics, is_best=False)
                print(f"Visualizations complete for epoch {epoch + 1}.")

            if is_best:
                print(f"Saving best checkpoint for epoch {epoch + 1}...")
                self.save_checkpoint(epoch + 1, metrics_dict, is_best=True)
                print(f"Best checkpoint saved. Generating visualizations...")
                self.save_visualizations(epoch + 1, val_metrics, is_best=True)
                print(f"Visualizations complete for best checkpoint.")
            if self.patience >= 1 and self.patience_counter >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                # Save final checkpoint if not already saved
                if not should_checkpoint and not is_best:
                    self.save_checkpoint(epoch + 1, {'train': train_metrics, 'val': val_metrics}, is_best=False)
                    self.save_visualizations(epoch + 1, val_metrics, is_best=False)
                break
        print(f"\nTraining completed. Best F1-score: {self.best_f1:.4f}")
        test_metrics = self.test_model()
        print(f"Test Results - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")
        self.metrics_tracker.save_final_results(self.run_dir / 'training_stats.json')
        final_vis_dir = self.run_dir / "final_results"
        final_vis_dir.mkdir(exist_ok=True)
        create_confusion_matrix(test_metrics['targets'], test_metrics['predictions'],
                               self.train_dataset.classes, final_vis_dir / 'test_confusion_matrix.png')
        create_per_class_metrics(test_metrics['targets'], test_metrics['predictions'],
                                 self.train_dataset.classes, final_vis_dir / 'test_per_class_metrics.png')

def main():
    parser = argparse.ArgumentParser(description='Train driver distraction classification model')
    parser.add_argument('config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--dataset-path', type=str, default=None, 
                       help='Path to camera directory (e.g., Dataset/Cam2 or Dataset/Cam4) containing split files')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU usage instead of GPU (useful for testing)')
    args = parser.parse_args()
    
    trainer = DistractionTrainer(args.config, args.dataset_path, args.cpu)
    trainer.train()

if __name__ == '__main__':
    main()