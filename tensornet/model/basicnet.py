import torch.nn as nn

from tensornet.model.base_model import BaseModel


class BasicNet(BaseModel):

    def __init__(self, num_classes, dropout_rate=0.0, in_channels=3):
        """This function instantiates all the model layers."""

        super(BasicNet, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Input: 32x32x32 | Output: 32x32x64 | RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Input: 32x32x64 | Output: 16x16x64 | RF: 6x6
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  # Input: 16x16x64 | Output: 16x16x32 | RF: 6x6
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # Input: 16x16x32 | Output: 16x16x32 | RF: 10x10
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Input: 16x16x32 | Output: 16x16x64 | RF: 14x14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Input: 16x16x64 | Output: 8x8x64 | RF: 16x16
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  # Input: 8x8x64 | Output: 8x8x32 | RF: 16x16
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # Input: 8x8x32 | Output: 8x8x32 | RF: 24x24
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            # Depthwise separable convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=1),  # Input: 8x8x32 | Output: 8x8x32 | RF: 32x32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),  # Input: 8x8x32 | Output: 8x8x64 | RF: 32x32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock3 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Input: 8x8x64 | Output: 4x4x64 | RF: 36x36
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  # Input: 4x4x64 | Output: 4x4x32 | RF: 36x36
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # Input: 4x4x32 | Output: 4x4x32 | RF: 52x52
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            # Dilated convolution
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, dilation=2),  # Input: 4x4x32 | Output: 4x4x64 | RF: 84x84
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )  # Input: 4x4x64 | Output: 1x1x64 | RF: 108x108

        self.fc = nn.Sequential(
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """This function defines the forward pass of the model.

        Args:
            x: Input.
        
        Returns:
            Model output.
        """

        x = self.convblock1(x)
        x = self.transblock1(x)
        x = self.convblock2(x)
        x = self.transblock2(x)
        x = self.convblock3(x)
        x = self.transblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x
