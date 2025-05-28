import torch
import torch.nn as nn


class PixelReconstructionLoss(nn.Module):
    def __init__(self):
        super(PixelReconstructionLoss, self).__init__()

    def forward(self, preds, gt, mask=None):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(preds[mask], gt[mask])
        return loss
