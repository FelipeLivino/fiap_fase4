"""Standard baseline CNN architecture for CardioIA Vision."""

from __future__ import annotations

import torch
from torch import nn


class StandardBaselineCNN(nn.Module):
    """Simple AlexNet-inspired CNN baseline.

    This model is intentionally simpler than `CardioIACustomCNN`. It provides a
    fair baseline to check whether the custom architecture adds value.

    Output:
        A single logit per image. Use with `nn.BCEWithLogitsLoss`.
    """

    def __init__(self, dropout: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)


def create_standard_cnn(dropout: float = 0.30) -> StandardBaselineCNN:
    """Factory used by notebooks and training scripts."""
    return StandardBaselineCNN(dropout=dropout)
