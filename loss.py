import torch
import torch.nn as nn

from tensornet.model.loss import ssim_loss, dice_loss, rmse_loss, bce_loss, bce_dice_loss


class SsimDiceLoss(nn.Module):
    
    def __init__(self):
        super(SsimDiceLoss, self).__init__()
        self.ssim = ssim_loss(channel=1)
        self.dice = dice_loss()

    def forward(self, prediction, label):
        return (
            2 * self.ssim(torch.sigmoid(prediction[0]), label[0]) +
            self.dice(torch.sigmoid(prediction[1]), label[1])
        )


class RmseBceDiceLoss(nn.Module):
    
    def __init__(self):
        super(RmseBceDiceLoss, self).__init__()
        self.rmse = rmse_loss()
        self.bce_dice = bce_dice_loss()

    def forward(self, prediction, label):
        return (
            2 * self.rmse(torch.sigmoid(prediction[0]), label[0]) +
            self.bce_dice(prediction[1], label[1])
        )


class RmseBceLoss(nn.Module):
    
    def __init__(self):
        super(RmseBceLoss, self).__init__()
        self.rmse = rmse_loss()
        self.bce = bce_loss()

    def forward(self, prediction, label):
        return (
            2 * self.rmse(torch.sigmoid(prediction[0]), label[0]) +
            self.bce(prediction[1], label[1])
        )
