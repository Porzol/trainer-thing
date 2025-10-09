import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3Model(nn.Module):
    def __init__(self, num_classes, dropout=0.5, activation='relu'):
        super(MobileNetV3Model, self).__init__()
        
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, 1280),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(1280, 512),
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