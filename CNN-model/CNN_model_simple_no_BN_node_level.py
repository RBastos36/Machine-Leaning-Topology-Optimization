import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # Removed MaxPool2d to preserve spatial dimensions

    def forward(self, x):
        return self.conv(x)


class TopologyOptimizationCNN(nn.Module):
    def __init__(self, input_height=180, input_width=60):
        super(TopologyOptimizationCNN, self).__init__()

        # Encoder path
        self.encoder = nn.Sequential(
            ConvBlock(5, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128)
        )

        # Decoder path to maintain spatial dimensions
        self.decoder = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, 2, kernel_size=1)  # Output 2 channels: x and y displacements
        )

    def forward(self, x):
        # x shape: [batch_size, 5, height, width]
        features = self.encoder(x)
        output = self.decoder(features)
        # output shape: [batch_size, 2, height, width]
        return output