import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out


class TopologyOptimizationCNN(nn.Module):
    def __init__(self):
        super(TopologyOptimizationCNN, self).__init__()

        # Encoder with Residual Blocks
        self.encoder = nn.Sequential(
            ResidualBlock(5, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 2, kernel_size=1)  # Predict X, Y displacements
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
