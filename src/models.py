
"""
ResNet18 model for real/fake image classification.
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Classifier(nn.Module):
    """ResNet18 pretrained on ImageNet, adapted to binary classification."""

    def __init__(self):
        super().__init__()

        # Load ResNet18 with pretrained ImageNet weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the original classification layer (1000 classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        """Return the logits for the two classes."""
        return self.model(x)
