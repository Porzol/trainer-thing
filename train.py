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
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
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
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
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
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
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
        
    def _export_onnx(self, onnx_path):
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.model.eval()
        torch.onnx.export(self.model, dummy_input, onnx_path, export_params=True, opset_version=11,
                         do_constant_folding=True, input_names=['input'], output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
    def save_visualizations(self, epoch, val_metrics, is_best=False):
        vis_dir = self.run_dir / ("best" if is_best else f"epoch_{epoch:02d}") / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        create_confusion_matrix(val_metrics['targets'], val_metrics['predictions'],
                               self.train_dataset.classes, vis_dir / 'confusion_matrix.png')
        create_per_class_metrics(val_metrics['targets'], val_metrics['predictions'],
                                 self.train_dataset.classes, vis_dir / 'per_class_metrics.png')
        self.metrics_tracker.save_metrics_plot(vis_dir / 'metrics_history.png')
        save_sample_predictions(self.model, self.test_loader, self.device,
                               self.train_dataset.classes, vis_dir / 'sample_predictions.png')
        
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
            if should_checkpoint or is_best:
                self.save_checkpoint(epoch + 1, {'train': train_metrics, 'val': val_metrics}, is_best)
                self.save_visualizations(epoch + 1, val_metrics, is_best)
            if self.patience_counter >= self.config['patience']:
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