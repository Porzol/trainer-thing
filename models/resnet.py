import torch.nn as nn
import torchvision.models as models
from .activations import get_activation

class ResNetV2Model(nn.Module):
    def __init__(self, num_classes, dropout=0.5, activation='relu'):
        super(ResNetV2Model, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)