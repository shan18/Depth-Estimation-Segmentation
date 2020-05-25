import torch
import torch.nn as nn
import torch.nn.functional as F

from tensornet.model.base_model import BaseModel
from tensornet.model.resnet import BasicBlock


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, res_block=None):
        super(Block, self).__init__()

        self.layer = self._make_layer(in_planes, planes)
        self.res_block = None
        if not res_block is None:
            self.res_block = nn.Sequential(
                res_block(planes, planes)
            )
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """This function defines the forward pass of the block.

        Args:
            x: Input.
        
        Returns:
            Model output.
        """
        x = self.layer(x)
        if not self.res_block is None:
            x = x + self.res_block(x)
        return x


class ResModel(BaseModel):

    def __init__(self, block, res_block, num_classes, in_channels=3):
        super(ResModel, self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block_layers = nn.Sequential(
            block(64, 128, res_block=res_block),
            block(128, 256),
            block(256, 512, res_block=res_block)
        )
        
        self.pool = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """This function defines the forward pass of the model.

        Args:
            x: Input.
        
        Returns:
            Model output.
        """
        x = self.prep_layer(x)
        x = self.block_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def ResidualModel(num_classes, in_channels=3):
    return ResModel(ResidualBlock, BasicBlock, num_classes, in_channels=in_channels)
