"""
Model components for 2D prostate segmentation on HipMRI slices.

UNet: baseline 2D U-Net (Ronneberger et al., 2015, "U-Net: Convolutional
Networks for Biomedical Image Segmentation"). Used as the benchmark that the
Improved U-Net is compared against on the same splits.

Only PyTorch is used here (no NumPy), as required by the assessment spec.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder step: 2x2 max-pool to halve resolution, then DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Decoder step: transposed conv to double resolution, concatenate the
    matching encoder features (skip connection), then DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Baseline 2D U-Net.

    Four down/up levels, so input height and width must be divisible by 16
    (HipMRI slices are 256x128). Outputs raw logits (B, num_classes, H, W);
    softmax is applied in the loss and at inference.

    `base_channels` sets the width of the first level and doubles each level.
    The original paper uses 64; 32 keeps the model light enough for a laptop.
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape check: one forward and backward pass on a dummy batch.
    model = UNet()
    x = torch.randn(2, 1, 256, 128)
    out = model(x)
    out.mean().backward()
    print(f"UNet output {tuple(out.shape)}, {count_parameters(model):,} parameters")
