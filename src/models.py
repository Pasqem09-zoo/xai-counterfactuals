
"""
ResNet18 model for real/fake image classification.
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

from src.config import DROPOUT


class ResNet18Classifier(nn.Module):
    """ResNet18 pretrained on ImageNet, adapted to binary classification."""

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the original classification layer
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        """Return the logits for the two classes."""
        return self.model(x)






class ResNet50Classifier(nn.Module):
    """ResNet50 pretrained on ImageNet for real/fake classification."""

    def __init__(self):
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.model(x)
