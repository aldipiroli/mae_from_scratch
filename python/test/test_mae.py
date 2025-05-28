import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mae import PatchImage, MAE
import torch


def test_path_image():
    patch_image = PatchImage()
    b, c, h, w = 8, 3, 64, 64
    x = torch.randn(b, c, h, w)
    y = patch_image(x)
    y_reshape = patch_image.fold(y, output_size=(h, w))
    assert torch.equal(y_reshape, x)


def test_unshuffle_tokes():
    mae = MAE()

    b, n, embed_size = 8, 64, 256
    x = torch.randn(b, n, embed_size)
    random_indices = torch.stack([torch.randperm(n) for _ in range(b)], dim=0)  # (b, n)
    random_indices = random_indices.unsqueeze(-1).expand(-1, -1, embed_size)
    x_shuffle = torch.gather(x, 1, random_indices)

    x_unshuffle = mae.unshuffle_tokens(x_shuffle, random_indices)
    assert torch.equal(x_unshuffle, x)


def test_mae_forward_pass():
    mae = MAE()
    b, c, h, w = 8, 3, 64, 64
    x = torch.randn(b, c, h, w)
    output_dict = mae(x)
    assert output_dict["pixel_preds"].shape == (b, c, h, w)
    assert output_dict["pred_token_mask"].shape == (b, c, h, w)


if __name__ == "__main__":
    test_path_image()
    test_unshuffle_tokes()
    test_mae_forward_pass()
    print("Test passed.")
