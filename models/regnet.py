import torch
import torch.nn as nn
import torchvision.models as models

class RegNetModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5, activation='relu'):
        super(RegNetModel, self).__init__()
        
        self.backbone = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
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