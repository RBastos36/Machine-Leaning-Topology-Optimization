# Author: Ricardo A. O. Bastos
# Created: June 2025


import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)

class TopologyOptimizationCNN(nn.Module):
    def __init__(self):
        super(TopologyOptimizationCNN, self).__init__()

        # 5 total channels across 3 matrices:
        # 1st matrix: domain (1 channel)
        # 2nd matrix: loads (2 channels - x and y directions)
        # 3rd matrix: constraints (2 channels - x and y directions)
        self.features = nn.Sequential(
            ConvBlock(5, 32),
            ConvBlock(32, 64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = self._build_classifier()

    def _build_classifier(self):
        # Test input now has 5 channels across 3 matrices
        test_input = torch.zeros(1, 5, 180, 60)
        with torch.no_grad():
            feature_test = self.features(test_input)
            flattened_size = feature_test.view(1, -1).size(1)

        return nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)