import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def create_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _add_bar_labels(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

def create_per_class_metrics(y_true, y_pred, class_names, save_path):
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width/2, precision_per_class, width, label='Precision')
    rects2 = ax.bar(x + width/2, recall_per_class, width, label='Recall')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _add_bar_labels(ax, rects1)
    _add_bar_labels(ax, rects2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _denormalize_image(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(img * std + mean, 0, 1)

def _collect_predictions(model, test_loader, device, num_samples):
    images, predictions, true_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for i in range(min(len(data), num_samples - len(images))):
                img = _denormalize_image(data[i].cpu())
                images.append(img.permute(1, 2, 0).numpy())
                predictions.append(pred[i].cpu().item())
                true_labels.append(target[i].cpu().item())
                if len(images) >= num_samples:
                    break
            if len(images) >= num_samples:
                break
    return images, predictions, true_labels

def _get_class_name(class_idx, class_names):
    return class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"

def save_sample_predictions(model, test_loader, device, class_names, save_path, num_samples=16):
    images, predictions, true_labels = _collect_predictions(model, test_loader, device, num_samples)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    for i in range(min(len(images), num_samples)):
        axes[i].imshow(images[i])
        pred_class = _get_class_name(predictions[i], class_names)
        true_class = _get_class_name(true_labels[i], class_names)
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        axes[i].set_title(f'Pred: {pred_class}\nTrue: {true_class}', color=color, fontsize=10)
        axes[i].axis('off')
    for i in range(len(images), num_samples):
        axes[i].axis('off')
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()