import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class DoubleConvBlock(BaseModel):

    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ResEncoderBlock(BaseModel):

    def __init__(self, in_channels, out_channels):
        super(ResEncoderBlock, self).__init__()

        self.double_conv = DoubleConvBlock(
            in_channels, out_channels
        )
        self.skip_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
        self.down = nn.MaxPool2d(2)
    
    def forward(self, x):
        identity = self.skip_conv(x)
        out = self.double_conv(x)
        out = out + identity
        return self.down(out), out


class ResDecoderBlock(BaseModel):

    def __init__(self, in_channels, out_channels):
        super(ResDecoderBlock, self).__init__()

        self.transition_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        )
        self.enc_skip_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
        self.skip_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
        self.double_conv = DoubleConvBlock(
            in_channels, out_channels
        )

    def forward(self, x, encoder_input, skip_input=None):
        # Transition
        x = self.transition_conv(x)
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False
        )
        if not skip_input is None:
            encoder_input = torch.cat(
                [encoder_input, skip_input], dim=1
            )
            encoder_input = self.enc_skip_conv(encoder_input)
        x = torch.cat([x, encoder_input], dim=1)

        # Decoding
        identity = self.skip_conv(x)
        out = self.double_conv(x)
        out = out + identity
        return out


class DSResNet(BaseModel):

    def __init__(self):
        super(DSResNet, self).__init__()

        # Encoder Network
        # ===============

        # Preparation Block for bg
        self.b1 = ResEncoderBlock(3, 16)
        self.b2 = ResEncoderBlock(16, 32)

        # Preparation Block for bg_fg
        self.bf1 = ResEncoderBlock(3, 16)
        self.bf2 = ResEncoderBlock(16, 32)

        # Join both inputs
        self.merge = nn.Conv2d(64, 32, kernel_size=1)

        # Merged encoder network
        self.enc1 = ResEncoderBlock(32, 64)
        self.enc2 = ResEncoderBlock(64, 128)
        self.enc3 = ResEncoderBlock(128, 256)
        self.enc4 = ResEncoderBlock(256, 512)

        # Decoder Network
        # ===============

        # Decoder Network - Segmentation
        self.Mdec3 = ResDecoderBlock(512, 256)
        self.Mdec2 = ResDecoderBlock(256, 128)
        self.Mdec1 = ResDecoderBlock(128, 64)
        self.M2 = ResDecoderBlock(64, 32)
        self.M1 = ResDecoderBlock(32, 16)
        self.M0 = nn.Conv2d(16, 1, kernel_size=1)

        # Decoder Network - Depth
        self.Ddec3 = ResDecoderBlock(512, 256)
        self.Ddec2 = ResDecoderBlock(256, 128)
        self.Ddec1 = ResDecoderBlock(128, 64)
        self.D2 = ResDecoderBlock(64, 32)
        self.D1 = ResDecoderBlock(32, 16)
        self.D0 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # bg
        b1_down, b1 = self.b1(x['bg'])
        b2_down, b2 = self.b2(b1_down)

        # bg_fg
        bf1_down, bf1 = self.bf1(x['bg_fg'])
        bf2_down, bf2 = self.bf2(bf1_down)

        # Merging
        merge = torch.cat([b2_down, bf2_down], dim=1)
        merge = self.merge(merge)

        # Merged Encoder
        enc1_down, enc1 = self.enc1(merge)
        enc2_down, enc2 = self.enc2(enc1_down)
        enc3_down, enc3 = self.enc3(enc2_down)
        _, enc4 = self.enc4(enc3_down)

        # Decoder - Segmentation
        Mdec3 = self.Mdec3(enc4, enc3)
        Mdec2 = self.Mdec2(Mdec3, enc2)
        Mdec1 = self.Mdec1(Mdec2, enc1)
        m2 = self.M2(Mdec1, b2, bf2)
        m1 = self.M1(m2, b1, bf1)
        outM = self.M0(m1)

        # Decoder - Depth
        Ddec3 = self.Ddec3(enc4, enc3)
        Ddec2 = self.Ddec2(Ddec3, enc2)
        Ddec1 = self.Ddec1(Ddec2, enc1)
        d2 = self.D2(Ddec1, b2, bf2)
        d1 = self.D1(d2, b1, bf1)
        outD = self.D0(d1)

        return outD, outM
