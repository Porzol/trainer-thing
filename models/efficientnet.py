import torch.nn as nn
import torchvision.models as models
from .activations import get_activation

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5, activation='relu'):
        super(EfficientNetModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)