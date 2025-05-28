import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.loss_functions import PixelReconstructionLoss

def test_pixel_reconstruction_loss():
    loss_function = PixelReconstructionLoss()
    preds = torch.randn(8, 3, 64, 64)
    gt = torch.randn(8, 3, 64, 64)
    loss = loss_function(preds, gt)
    assert loss > 0
