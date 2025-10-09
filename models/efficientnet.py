import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5, activation='relu'):
        super(EfficientNetModel, self).__init__()
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.classifier[1].in_features, 512),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        return self.backbone(x)