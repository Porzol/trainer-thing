import torch
import torch.nn as nn
import torchvision.models as models

class ResNetV2Model(nn.Module):
    def __init__(self, num_classes, dropout=0.5, activation='relu'):
        super(ResNetV2Model, self).__init__()
        
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
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