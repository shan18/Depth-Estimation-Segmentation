import os
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from tensornet.data.utils import to_numpy


class TensorBoard:

    def __init__(self, logdir=None, images=None, device='cpu'):
        """Setup Tensorboard.

        Args:
            logdir (str, optional): Save directory location.
                Default is runs/CURRENT_DATETIME_HOSTNAME,
                which changes after each run. (default: None)
            images (torch.Tensor, optional): Batch of images for
                which predictions will be done. (default: None)
            device (str or torch.device, optional): Device where the data
                will be loaded. (default='cpu')
        """
        self.logdir = logdir
        self.images = images
        self.device = device
        self.writer = SummaryWriter(self.logdir)

        # Create directory for storing image predicitons
        self.img_dir = os.path.join(self.writer.log_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)

        if not (self.device == 'cpu' or self.images is None):
            self._move_images()
    
    def _move_images(self):
        """Move images to a device."""
        if isinstance(self.images, dict):
            for key in self.images:
                self.images[key] = self.images[key].to(self.device)
        elif isinstance(self.images, (list, tuple)):
            images = []
            for data in self.images:
                images.append(data.to(self.device))
            self.images = images
        else:
            self.images = self.images.to(self.device)
    
    def write_model(self, model):
        """Write graph to tensorboard.

        Args:
            model (torch.nn.Module): Model Instance.
        """
        if not self.images is None:
            self.writer.add_graph(model, self.images)
    
    def write_image(self, image, image_name):
        image = image.detach().cpu()
        image_grid = make_grid(image)

        self.writer.add_image(image_name, image_grid)  # Write summary

        with open(
            os.path.join(self.img_dir, f'{image_name}.jpeg'),
            'wb'
        ) as fimg: # Save predictions
            save_image(image_grid, fimg)
    
    def write_images(self, model, activation_fn=None, image_name=None):
        """Write images to tensorboard.

        Args:
            model (torch.nn.Module): Model Instance.
            activation_fn (optional): Activation function to apply on
                model outputs. (default: None)
            image_name (str, optional): Name of the image to be written.
                (default: None)
        """
        if image_name is None:
            image_name = 'model_predictions'
        
        model.eval()
        predictions = model(self.images)
        if not activation_fn is None:
            predictions = activation_fn(predictions)
        
        if isinstance(predictions, (tuple, list)):
            for idx, prediction in enumerate(predictions):
                self.write_image(prediction, f'{idx}_{image_name}')
        else:
            self.write_image(predictions, image_name)
    
    def write_scalar(self, scalar, value, step_value):
        """Write scalar metrics to tensorboard.

        Args:
            scalar (str): Data identifier.
            value (float or string/blobname): Value to save.
            step_value (int): Global step value to record.
        """
        self.writer.add_scalar(scalar, value, step_value)
