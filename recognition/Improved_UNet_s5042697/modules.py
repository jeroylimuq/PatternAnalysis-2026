"""
Model components for 2D prostate segmentation on HipMRI slices.

UNet: baseline 2D U-Net (Ronneberger et al., 2015, "U-Net: Convolutional
Networks for Biomedical Image Segmentation"). Used as the benchmark that the
Improved U-Net is compared against on the same splits.

ImprovedUNet: 2D adaptation of the 3D network in Isensee et al., 2017,
"Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to
the BRATS 2017 Challenge". Differences from the baseline: pre-activation
residual context modules with dropout, strided-conv downsampling, instance
norm with leaky ReLU, localization modules in the decoder, and deep
supervision that sums segmentation maps from three decoder levels.

CAN: 2D Context Aggregation Network (Yu and Koltun, 2016, "Multi-scale
context aggregation by dilated convolutions"). Instead of shrinking and
re-growing the image like a U-Net, it stays at full resolution and widens
its view with dilated 3x3 convolutions whose dilation doubles each layer.

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


def norm_act(channels):
    """Instance norm followed by leaky ReLU (slope 1e-2), as in Isensee et al."""
    return nn.Sequential(
        nn.InstanceNorm2d(channels, affine=True),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )


class ContextModule(nn.Module):
    """
    Pre-activation residual block: (norm, act, 3x3 conv), dropout,
    (norm, act, 3x3 conv). The residual sum with the input is done by the
    caller, so the same module can follow either a stride-1 or stride-2 conv.
    """

    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            norm_act(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout2d(dropout),
            norm_act(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class EncoderLevel(nn.Module):
    """
    One encoder level: 3x3 conv (stride 2 to downsample, stride 1 on the
    first level) followed by a context module, joined by a residual sum.
    """

    def __init__(self, in_channels, out_channels, stride, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.context = ContextModule(out_channels, dropout)

    def forward(self, x):
        x = self.conv(x)
        return x + self.context(x)


class UpsamplingModule(nn.Module):
    """Double the resolution, then a 3x3 conv that halves the channels."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_act(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class LocalizationModule(nn.Module):
    """
    Applied after concatenating a skip connection: 3x3 conv to recombine the
    features, then 1x1 conv that halves the channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            norm_act(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            norm_act(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ImprovedUNet(nn.Module):
    """
    2D Improved U-Net (Isensee et al., 2017).

    Five encoder levels with base_channels * (1, 2, 4, 8, 16) features, so
    input height and width must be divisible by 16. Outputs raw logits
    (B, num_classes, H, W).

    Deep supervision: 1x1 segmentation layers on the decoder outputs at 1/4,
    1/2 and full resolution. Coarser maps are upsampled and added to the next
    finer one, so every decoder level gets a direct gradient from the loss.
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=16, dropout=0.3):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = EncoderLevel(in_channels, c, stride=1, dropout=dropout)
        self.enc2 = EncoderLevel(c, c * 2, stride=2, dropout=dropout)
        self.enc3 = EncoderLevel(c * 2, c * 4, stride=2, dropout=dropout)
        self.enc4 = EncoderLevel(c * 4, c * 8, stride=2, dropout=dropout)
        self.enc5 = EncoderLevel(c * 8, c * 16, stride=2, dropout=dropout)

        # Decoder: upsample, concat skip, localize
        self.up4 = UpsamplingModule(c * 16, c * 8)
        self.loc4 = LocalizationModule(c * 16, c * 8)
        self.up3 = UpsamplingModule(c * 8, c * 4)
        self.loc3 = LocalizationModule(c * 8, c * 4)
        self.up2 = UpsamplingModule(c * 4, c * 2)
        self.loc2 = LocalizationModule(c * 4, c * 2)
        self.up1 = UpsamplingModule(c * 2, c)
        self.final_conv = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1),
            norm_act(c * 2),
        )

        # Deep supervision segmentation layers
        self.seg3 = nn.Conv2d(c * 4, num_classes, kernel_size=1)
        self.seg2 = nn.Conv2d(c * 2, num_classes, kernel_size=1)
        self.seg1 = nn.Conv2d(c * 2, num_classes, kernel_size=1)
        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d4 = self.loc4(torch.cat([x4, self.up4(x5)], dim=1))
        d3 = self.loc3(torch.cat([x3, self.up3(d4)], dim=1))
        d2 = self.loc2(torch.cat([x2, self.up2(d3)], dim=1))
        d1 = self.final_conv(torch.cat([x1, self.up1(d2)], dim=1))

        seg = self.upscale(self.seg3(d3)) + self.seg2(d2)
        return self.upscale(seg) + self.seg1(d1)


class DilatedConv(nn.Module):
    """3x3 convolution with the given dilation, then batch norm and ReLU."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CAN(nn.Module):
    """
    2D Context Aggregation Network (Yu and Koltun, 2016).

    A front-end of two ordinary 3x3 convolutions extracts local features,
    then the context module stacks 3x3 convolutions with dilations
    (1, 1, 2, 4, 8, 16, 32, 1). Doubling the dilation grows the receptive
    field exponentially while the feature maps stay at full 256x128
    resolution, so no detail is lost to downsampling. The paper's context
    module ends at dilation 16 for its image size; dilation 32 is added so
    the receptive field (about 135 px) spans the prostate's surroundings in
    these slices. A 1x1 convolution gives per-pixel class logits.

    Batch norm and ReLU match the baseline U-Net, so the comparison isolates
    dilated full-resolution context versus encoder-decoder downsampling.
    """

    DILATIONS = (1, 1, 2, 4, 8, 16, 32, 1)

    def __init__(self, in_channels=1, num_classes=6, base_channels=64):
        super().__init__()
        c = base_channels
        self.front_end = nn.Sequential(
            DilatedConv(in_channels, c, dilation=1),
            DilatedConv(c, c, dilation=1),
        )
        self.context = nn.Sequential(*[DilatedConv(c, c, d) for d in self.DILATIONS])
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.context(self.front_end(x)))


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape check: one forward and backward pass on a dummy batch.
    x = torch.randn(2, 1, 256, 128)
    for model in (UNet(), ImprovedUNet(), CAN()):
        out = model(x)
        out.mean().backward()
        print(f"{type(model).__name__} output {tuple(out.shape)}, "
              f"{count_parameters(model):,} parameters")
