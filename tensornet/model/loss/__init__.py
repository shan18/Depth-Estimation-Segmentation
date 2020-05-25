import torch.nn as nn

from .ssim import SSIMLoss, MSSSIMLoss
from .dice import DiceLoss, BCEDiceLoss
from .rmse import RMSELoss


def cross_entropy_loss():
    """Create Cross Entropy Loss.
    The loss automatically applies the softmax activation
    function on the prediction input.

    Returns:
        Cross entroy loss function
    """
    return nn.CrossEntropyLoss()


def bce_loss():
    """Create Binary Cross Entropy Loss.
    The loss automatically applies the sigmoid activation
    function on the prediction input.

    Returns:
        Binary cross entropy loss function
    """
    return nn.BCEWithLogitsLoss()


def mse_loss():
    """Create Mean Squared Error Loss.

    Returns:
        Mean squared error loss function
    """
    return nn.MSELoss()


def rmse_loss(smooth=1e-6):
    """Create Root Mean Squared Error Loss.

    Returns:
        Root mean squared error loss function
    """
    return RMSELoss(smooth=1e-6)


def ssim_loss(data_range=1.0, size_average=True, channel=1):
    """Create SSIM Loss.

    Args:
        data_range (float or int, optional): Value range of input
            images (usually 1.0 or 255). (default: 255)
        size_average (bool, optional): If size_average=True, ssim
            of all images will be averaged as a scalar. (default: True)
        channel (int, optional): input channels (default: 1)

    Returns:
        SSIM loss function
    """
    return SSIMLoss(
        data_range=data_range, size_average=size_average, channel=channel
    )


def ms_ssim_loss(data_range=1.0, size_average=True, channel=1):
    """Create MS-SSIM Loss.

    Args:
        data_range (float or int, optional): Value range of input
            images (usually 1.0 or 255). (default: 1.0)
        size_average (bool, optional): If size_average=True, ssim
            of all images will be averaged as a scalar. (default: True)
        channel (int, optional): input channels (default: 1)

    Returns:
        MS-SSIM loss function
    """
    return MSSSIMLoss(
        data_range=data_range, size_average=size_average, channel=channel
    )


def dice_loss(smooth=1):
    """Create Dice Loss.

    Args:
        smooth (float, optional): Smoothing value. A larger
            smooth value (also known as Laplace smooth, or
            Additive smooth) can be used to avoid overfitting.
            (default: 1)
    
    Returns:
        Dice loss function
    """
    return DiceLoss(smooth=smooth)


def bce_dice_loss(smooth=1e-6):
    """Create BCEDice Loss.

    Args:
        smooth (float, optional): Smoothing value.
    
    Returns:
        BCEDice loss function
    """
    return BCEDiceLoss(smooth=smooth)
