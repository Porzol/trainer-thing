import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score

def create_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_sample_predictions(model, test_loader, device, class_names, save_path, num_samples=16):
    model.eval()
    
    images = []
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            for i in range(min(len(data), num_samples - len(images))):
                img = data[i].cpu()
                img = (img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + 
                       torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
                img = torch.clamp(img, 0, 1)
                
                images.append(img.permute(1, 2, 0).numpy())
                predictions.append(pred[i].cpu().item())
                true_labels.append(target[i].cpu().item())
                
                if len(images) >= num_samples:
                    break
            
            if len(images) >= num_samples:
                break
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i in range(min(len(images), num_samples)):
        axes[i].imshow(images[i])
        
        pred_class = class_names[predictions[i]] if predictions[i] < len(class_names) else f"Class_{predictions[i]}"
        true_class = class_names[true_labels[i]] if true_labels[i] < len(class_names) else f"Class_{true_labels[i]}"
        
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        axes[i].set_title(f'Pred: {pred_class}\nTrue: {true_class}', color=color, fontsize=10)
        axes[i].axis('off')
    
    for i in range(len(images), num_samples):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()