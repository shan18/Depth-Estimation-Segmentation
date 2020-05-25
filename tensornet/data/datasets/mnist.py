import os
import numpy as np
from torchvision import datasets

from tensornet.data.datasets.dataset import BaseDataset


class MNIST(BaseDataset):
    """Load MNIST Dataset."""
    
    def _download(self, train=True, apply_transform=True):
        """Download dataset.

        Args:
            train (bool, optional): True for training data.
                (default: True)
            apply_transform (bool, optional): True if transform
                is to be applied on the dataset. (default: True)
        
        Returns:
            Downloaded dataset.
        """
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return datasets.MNIST(
            self.path, train=train, download=True, transform=transform
        )
    
    def _get_image_size(self):
        """Return shape of data i.e. image size."""
        return (1, 28, 28)
    
    def _get_classes(self):
        """Return list of classes in the dataset."""
        return tuple([
            str(x) for x in range(10)
        ])
    
    def _get_mean(self):
        """Returns mean of the entire dataset."""
        return 0.1307
    
    def _get_std(self):
        """Returns standard deviation of the entire dataset."""
        return 0.3081
