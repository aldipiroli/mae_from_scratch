import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.loss_functions import PixelReconstructionLoss
import torch


def test_pixel_reconstruction_loss():
    b, c, h, w = 8, 3, 64, 64
    x = torch.randn(b, c, h, w)
    y = torch.randn(b, c, h, w)
    mask = torch.ones(b, c, h, w).bool()
    mask[:, :, :, :10] = 0
    loss_fn = PixelReconstructionLoss()
    loss = loss_fn(x, y, mask)
    assert loss > 0


if __name__ == "__main__":
    test_pixel_reconstruction_loss()
    print("Test passed.")
