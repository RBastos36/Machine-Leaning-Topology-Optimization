import torch
import torch.nn as nn


def crop_tensor(tensor, target_shape):
    """
    Crops a tensor to match the target spatial dimensions.
    """
    _, _, H, W = target_shape
    return tensor[:, :, :H, :W]


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Optional: Add Layer Normalization
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # Add dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.dropout(x)
        return x


class TopologyOptimizationCNN(nn.Module):
    def __init__(self):
        super(TopologyOptimizationCNN, self).__init__()

        # Encoder (dropout rate: 0.3)
        self.enc1 = UNetBlock(5, 32, dropout_rate=0.3)
        self.enc2 = UNetBlock(32, 64, dropout_rate=0.3)
        self.enc3 = UNetBlock(64, 128, dropout_rate=0.3)
        self.enc4 = UNetBlock(128, 256, dropout_rate=0.3)

        self.pool = nn.MaxPool2d(2, ceil_mode=True)  # Ensure even-sized outputs

        # Bottleneck (dropout rate: 0.5)
        self.bottleneck = UNetBlock(256, 512, dropout_rate=0.5)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, output_padding=1)
        self.dec4 = UNetBlock(512, 256, dropout_rate=0.2)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, output_padding=1)
        self.dec3 = UNetBlock(256, 128, dropout_rate=0.2)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=1)
        self.dec2 = UNetBlock(128, 64, dropout_rate=0.2)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, output_padding=1)
        self.dec1 = UNetBlock(64, 32, dropout_rate=0.2)

        # Output layer
        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        up4 = self.up4(bottleneck)
        up4 = crop_tensor(up4, enc4.shape)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))

        up3 = self.up3(dec4)
        up3 = crop_tensor(up3, enc3.shape)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.up2(dec3)
        up2 = crop_tensor(up2, enc2.shape)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        up1 = crop_tensor(up1, enc1.shape)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        out = self.out_conv(dec1)
        return out