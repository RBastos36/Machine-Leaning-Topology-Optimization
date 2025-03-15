import torch.nn as nn


class TopologyOptimizationCNN(nn.Module):
    def __init__(self):
        super(TopologyOptimizationCNN, self).__init__()

        # Encoder (Deeper with Dropout)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Extra layer
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 2, kernel_size=1)  # Predict X, Y displacements
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
