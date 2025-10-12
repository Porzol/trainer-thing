import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class MetricsTracker:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.training_times = []
        
    def update(self, epoch, train_metrics, val_metrics):
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        self.train_accuracies.append(train_metrics['accuracy'])
        self.val_accuracies.append(val_metrics['accuracy'])
        self.train_f1_scores.append(train_metrics['f1_score'])
        self.val_f1_scores.append(val_metrics['f1_score'])
        self.train_precisions.append(train_metrics['precision'])
        self.val_precisions.append(val_metrics['precision'])
        self.train_recalls.append(train_metrics['recall'])
        self.val_recalls.append(val_metrics['recall'])
        self.training_times.append(train_metrics['training_time'])
        
    def _plot_metric(self, ax, train_data, val_data, title, ylabel):
        ax.plot(self.epochs, train_data, label='Train', marker='o')
        ax.plot(self.epochs, val_data, label='Validation', marker='s')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    def save_metrics_plot(self, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        self._plot_metric(axes[0, 0], self.train_losses, self.val_losses, 'Loss', 'Loss')
        self._plot_metric(axes[0, 1], self.train_accuracies, self.val_accuracies, 'Accuracy', 'Accuracy')
        self._plot_metric(axes[0, 2], self.train_f1_scores, self.val_f1_scores, 'F1-Score', 'F1-Score')
        self._plot_metric(axes[1, 0], self.train_precisions, self.val_precisions, 'Precision', 'Precision')
        self._plot_metric(axes[1, 1], self.train_recalls, self.val_recalls, 'Recall', 'Recall')
        axes[1, 2].plot(self.epochs, self.training_times, label='Training Time', marker='o', color='green')
        axes[1, 2].set_title('Training Time per Epoch')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_final_results(self, save_path):
        results = {
            'epochs': self.epochs,
            'train_metrics': {
                'loss': self.train_losses,
                'accuracy': self.train_accuracies,
                'f1_score': self.train_f1_scores,
                'precision': self.train_precisions,
                'recall': self.train_recalls,
                'training_time': self.training_times
            },
            'val_metrics': {
                'loss': self.val_losses,
                'accuracy': self.val_accuracies,
                'f1_score': self.val_f1_scores,
                'precision': self.val_precisions,
                'recall': self.val_recalls
            },
            'summary': {
                'best_val_accuracy': max(self.val_accuracies) if self.val_accuracies else 0,
                'best_val_f1': max(self.val_f1_scores) if self.val_f1_scores else 0,
                'total_training_time': sum(self.training_times),
                'avg_epoch_time': np.mean(self.training_times) if self.training_times else 0
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)