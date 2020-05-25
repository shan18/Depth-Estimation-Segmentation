import torch
import torch.nn as nn

from tensornet.model.base_model import BaseModel


class MaskNet(BaseModel):

    def __init__(self, dropout_rate=0.0, in_channels=3):
        """This function instantiates all the model layers."""

        super(MaskNet, self).__init__()

        self.prep_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),
        )
        self.prep_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),
        )

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate),
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        """This function defines the forward pass of the model.

        Args:
            x: Input.

        Returns:
            Model output.
        """

        x1 = self.prep_block_1(x['bg'])
        x2 = self.prep_block_2(x['bg_fg'])
        out = torch.cat([x1, x2], dim=1)

        out = self.convblock1(out)
        out = self.convblock2(out)
        out = self.convblock3(out)

        return out

