import torch
import torch.nn as nn


def crop_tensor(tensor, target_shape):
    """
    Crops a tensor to match the target spatial dimensions.
    """
    _, _, H, W = target_shape
    return tensor[:, :, :H, :W]


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Optional: Add Layer Normalization
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class TopologyOptimizationCNN(nn.Module):
    def __init__(self):
        super(TopologyOptimizationCNN, self).__init__()

        # Encoder
        self.enc1 = UNetBlock(5, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)
        self.enc4 = UNetBlock(128, 256)

        self.downsample1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = UNetBlock(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 32)

        # Output layer
        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)  # Predict X, Y displacements

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.downsample1(enc1))
        enc3 = self.enc3(self.downsample2(enc2))
        enc4 = self.enc4(self.downsample3(enc3))

        bottleneck = self.bottleneck(enc4)

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
