import torch.nn as nn
import torchvision.models as models
from .activations import get_activation

class MobileNetV3Model(nn.Module):
    def __init__(self, num_classes, dropout=0.5, activation='relu'):
        super(MobileNetV3Model, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(1280, 512),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)