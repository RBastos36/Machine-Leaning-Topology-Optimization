import torch
import torch.nn as nn


def crop_tensor(tensor, target_shape):
    """Crop tensor to match target shape - handles cases where tensor might be smaller than target"""
    _, _, H_target, W_target = target_shape
    _, _, H, W = tensor.size()

    # If tensor is already the right size, return as is
    if H == H_target and W == W_target:
        return tensor

    # If tensor is smaller than target, we can't crop it properly
    if H < H_target or W < W_target:
        # Use interpolation to resize instead of cropping
        return torch.nn.functional.interpolate(
            tensor,
            size=(H_target, W_target),
            mode='bilinear',
            align_corners=False
        )

    # Normal cropping case
    delta_H = H - H_target
    delta_W = W - W_target
    start_H = delta_H // 2
    start_W = delta_W // 2
    return tensor[:, :, start_H:start_H + H_target, start_W:start_W + W_target]


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.relu = nn.ReLU()

        # Handle cases where out_channels < 8 for GroupNorm
        num_groups = min(8, out_channels)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class TopologyOptimizationCNN(nn.Module):
    def __init__(self):
        super(TopologyOptimizationCNN, self).__init__()

        # Hybrid encoder - your original design
        self.enc1 = UNetBlock(5, 32, padding=0)  # valid padding - reduces size
        self.enc2 = UNetBlock(32, 64, padding=0)  # valid padding - reduces size
        self.enc3 = UNetBlock(64, 128, padding=1)  # same padding - preserves size
        self.enc4 = UNetBlock(128, 256, padding=1)  # same padding - preserves size

        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        # Bottleneck
        self.bottleneck = UNetBlock(256, 512, padding=1)  # same padding

        # Decoder with same padding
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(512, 256, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256, 128, padding=1)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 32, padding=1)

        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        # Store original input size for final resize
        original_size = x.shape[2:]

        # Debug: print input size
        # print(f"Input size: {x.shape}")

        # Encoder path
        enc1 = self.enc1(x)  # Size reduces due to valid padding
        # print(f"enc1 size: {enc1.shape}")

        enc2 = self.enc2(self.pool(enc1))  # Size reduces further due to valid padding
        # print(f"enc2 size: {enc2.shape}")

        enc3 = self.enc3(self.pool(enc2))  # Same padding from here - size preserved (except pooling)
        # print(f"enc3 size: {enc3.shape}")

        enc4 = self.enc4(self.pool(enc3))  # Same padding - size preserved (except pooling)
        # print(f"enc4 size: {enc4.shape}")

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        # print(f"bottleneck size: {bottleneck.shape}")

        # Decoder path - use improved crop function that handles all cases
        up4 = self.up4(bottleneck)
        # print(f"up4 size: {up4.shape}")
        enc4_crop = crop_tensor(enc4, up4.shape)
        # print(f"enc4_crop size: {enc4_crop.shape}")
        dec4 = self.dec4(torch.cat([up4, enc4_crop], dim=1))

        up3 = self.up3(dec4)
        # print(f"up3 size: {up3.shape}")
        enc3_crop = crop_tensor(enc3, up3.shape)
        # print(f"enc3_crop size: {enc3_crop.shape}")
        dec3 = self.dec3(torch.cat([up3, enc3_crop], dim=1))

        up2 = self.up2(dec3)
        # print(f"up2 size: {up2.shape}")
        enc2_crop = crop_tensor(enc2, up2.shape)
        # print(f"enc2_crop size: {enc2_crop.shape}")
        dec2 = self.dec2(torch.cat([up2, enc2_crop], dim=1))

        up1 = self.up1(dec2)
        # print(f"up1 size: {up1.shape}")
        enc1_crop = crop_tensor(enc1, up1.shape)
        # print(f"enc1_crop size: {enc1_crop.shape}")
        dec1 = self.dec1(torch.cat([up1, enc1_crop], dim=1))

        out = self.out_conv(dec1)
        # print(f"output size before resize: {out.shape}")

        # Resize output to match original input size
        if out.shape[2:] != original_size:
            out = torch.nn.functional.interpolate(
                out,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )

        # print(f"final output size: {out.shape}")
        return out
