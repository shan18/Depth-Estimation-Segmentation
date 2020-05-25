import os
import zipfile
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from tensornet.data.datasets.dataset import BaseDataset


class MODESTMuseum(BaseDataset):
    """Load MODEST Museum Dataset."""
    
    def _split_data(self):
        """Split data into training and validation set."""

        self.input_types = ['bg', 'bg_fg']
        self.output_types = ['bg_fg_mask', 'bg_fg_depth_map']
        
        # Set training data
        self.train_transform = {
            img_type: self._transform(data_type=img_type)
            for img_type in self.input_types
        }
        for img_type in self.output_types:  # Outputs will not be normalized
            self.train_transform[img_type] = self._transform(
                data_type=img_type, normalize=False
            )
        self.train_data = self._download()
        self.classes = self._get_classes()

        # Set validation data
        self.val_transform = {
            img_type: self._transform(train=False, data_type=img_type)
            for img_type in self.input_types
        }
        for img_type in self.output_types:  # Outputs will not be normalized
            self.val_transform[img_type] = self._transform(
                train=False, data_type=img_type, normalize=False
            )
        self.val_data = self._download(train=False)

    def _download(self, train=True, apply_transform=True):
        """Fetch dataset.

        Args:
            train (bool, optional): True for training data.
                (default: True)
            apply_transform (bool, optional): True if transform
                is to be applied on the dataset. (default: True)
        
        Returns:
            Fetched dataset.
        """
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return MODESTMuseumDataset(
            self.path, train=train, train_split=self.train_split, transform=transform
        )
    
    def _get_image_size(self):
        """Return shape of data and targets i.e. image size."""
        return {
            'bg': (3, 224, 224),
            'bg_fg': (3, 224, 224),
            'bg_fg_mask': (1, 224, 224),
            'bg_fg_depth_map': (1, 224, 224),
        }
    
    def _get_mean(self):
        """Returns mean of the entire dataset."""
        return {
            'bg': (0.40086, 0.46599, 0.53281),
            'bg_fg': (0.41221, 0.47368, 0.53431),
            'bg_fg_mask': 0.05207,
            'bg_fg_depth_map': 0.2981,
        }
    
    def _get_std(self):
        """Returns standard deviation of the entire dataset."""
        return {
            'bg': (0.25451, 0.24249, 0.23615),
            'bg_fg': (0.25699, 0.24577, 0.24217),
            'bg_fg_mask': 0.21686,
            'bg_fg_depth_map': 0.11561,
        }


class MODESTMuseumDataset(Dataset):
    """Create MODEST Museum Dataset."""

    def __init__(self, path, train=True, train_split=0.7, random_seed=1, transform=None):
        """Initializes the dataset for loading.

        Args:
            path (str): Path where dataset zip file is present.
            train (bool, optional): True for training data. (default: True)
            train_split (float, optional): Fraction of dataset to assign
                for training. (default: 0.7)
            download (bool, optional): If True, dataset will be downloaded.
                (default: True)
            random_seed (int, optional): Random seed value. This is required
                for splitting the data into training and validation datasets.
                (default: 1)
            transform (dict, optional): Transformations to apply on the dataset.
                (default: None)
        """
        super(MODESTMuseumDataset, self).__init__()
        
        self.path = path
        self.train = train
        self.train_split = train_split
        self.transform = transform
        self._validate_params()

        self._fetch_data()

        self._image_indices = np.arange(len(self.data))

        np.random.seed(random_seed)
        np.random.shuffle(self._image_indices)

        split_idx = int(len(self._image_indices) * train_split)
        self._image_indices = self._image_indices[:split_idx] if train else self._image_indices[split_idx:]
    
    def __len__(self):
        """Returns length of the dataset."""
        return len(self._image_indices)
    
    def __getitem__(self, index):
        """Fetch an item from the dataset.

        Args:
            index (int): Index of the item to fetch.
        
        Returns:
            Input and their corresponding labels.
        """
        image_index = self._image_indices[index]
        
        # Input
        image_data = {
            img_type: Image.open(img_path)
            for img_type, img_path in self.data[image_index].items()
        }
        for img_type in image_data:
            if not self.transform[img_type] is None:
                image_data[img_type] = self.transform[img_type](image_data[img_type])
        
        # Target
        image_target = {
            img_type: Image.open(img_path)
            for img_type, img_path in self.targets[image_index].items()
        }
        for img_type in image_target:
            if not self.transform[img_type] is None:
                image_target[img_type] = self.transform[img_type](image_target[img_type])

        return image_data, image_target
    
    def _fetch_data(self):
        """Fetch the image paths of the downloaded dataset."""
        self.data, self.targets = [], []
        with open(os.path.join(self.path, 'file_map.txt')) as f:
            path_prefixes = ['bg', 'bg_fg', 'bg_fg_mask', 'bg_fg_depth_map']
            for line in f.readlines():
                imgs = [os.path.join(self.path, p, i + '.jpeg') for p, i in zip(path_prefixes, line[:-1].split('\t'))]
                self.data.append({p: i for p, i in zip(path_prefixes[:2], imgs[:2])})
                self.targets.append({p: i for p, i in zip(path_prefixes[2:], imgs[2:])})
    
    def __repr__(self):
        """Representation string for the dataset object."""
        head = 'Dataset MODEST Museum'
        body = ['Number of datapoints: {}'.format(self.__len__())]
        if self.path is not None:
            body.append('Root location: {}'.format(self.path))
        body += [f'Split: {"Train" if self.train else "Test"}']
        if hasattr(self, 'transforms') and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)
    
    def _validate_params(self):
        """Validate input parameters."""
        if self.train_split > 1:
            raise ValueError('train_split must be less than 1')
