import torch
import torch.nn as nn

class PixelReconstructionLoss(nn.Module):
    def __init__(self):
        super(PixelReconstructionLoss, self).__init__()

    def forward(self, preds, gt, mask, normalize=False):
        loss_fn = torch.nn.MSELoss()
        mean = torch.mean(gt, 2, keepdim=True)
        std = torch.std(gt, 2, keepdim=True) + 1e-6
        gt_norm = (gt - mean) / std
        if normalize:
            loss = loss_fn(preds[mask], gt_norm[mask])
        else:
            loss = loss_fn(preds[mask], gt[mask])
        return loss
