import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_tensor(tensor, target_tensor):
    """
    Crops a tensor to match the spatial dimensions of a target tensor.
    """
    target_size = target_tensor.size()[2:]
    tensor_size = tensor.size()[2:]

    delta_h = tensor_size[0] - target_size[0]
    delta_w = tensor_size[1] - target_size[1]

    if delta_h > 0 and delta_w > 0:
        return tensor[:, :, delta_h // 2:-(delta_h // 2 + delta_h % 2), delta_w // 2:-(delta_w // 2 + delta_w % 2)]
    else:
        return tensor


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)  # Valid padding
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)  # Valid padding
        self.relu = nn.ReLU()

        # Layer Normalization
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class TopologyOptimizationCNN(nn.Module):
    """
    A U-Net architecture with only 2 encoder/decoder blocks for 181x61 inputs.
    This version uses valid padding throughout the network.
    """

    def __init__(self):
        super(TopologyOptimizationCNN, self).__init__()

        # Encoder - ONLY 2 BLOCKS
        self.enc1 = UNetBlock(5, 32)
        self.enc2 = UNetBlock(32, 64)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(64, 128)

        # Decoder - ONLY 2 BLOCKS
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 32)

        # Output layer
        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        # Store original input size
        original_size = x.size()[2:]

        # Encoder path
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(pool2)

        # Decoder path
        up2 = self.up2(bottleneck)
        # Crop features for concatenation
        enc2_cropped = crop_tensor(enc2, up2)
        dec2 = self.dec2(torch.cat([up2, enc2_cropped], dim=1))

        up1 = self.up1(dec2)
        # Crop features for concatenation
        enc1_cropped = crop_tensor(enc1, up1)
        dec1 = self.dec1(torch.cat([up1, enc1_cropped], dim=1))

        # Output
        out = self.out_conv(dec1)

        # Ensure output has same dimensions as input
        if out.size()[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)

        return out