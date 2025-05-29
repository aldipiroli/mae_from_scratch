import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.mae import MAE, EmbedMasking, PatchImage


def test_patch_image():
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


def test_patch_and_unshuffle():
    b, c, h, w = 8, 3, 64, 64
    x = torch.randn(b, c, h, w)

    mae = MAE(embed_size=192)
    patch_image = PatchImage()
    x_patch = patch_image(x)

    embed_mask = EmbedMasking(mask_fraction=0.25)
    b, n, d = x_patch.shape
    x_embed, random_indices = embed_mask(x_patch)
    mask_tokens = torch.zeros(b, 48, d)

    pred_token_mask = torch.ones((b, n))
    pred_token_mask[:, : x_embed.shape[1]] = 0
    pred_token_mask = pred_token_mask.unsqueeze(-1).expand(-1, -1, d)

    x_encoder_w_masked_tokens = torch.cat([x_embed, mask_tokens], 1)

    x_unshuffle = mae.unshuffle_tokens(x_encoder_w_masked_tokens, random_indices)
    pred_token_mask_unshuffle = mae.unshuffle_tokens(pred_token_mask, random_indices)

    x_fold = patch_image.fold(x_unshuffle, (h, w))
    pred_token_mask_fold = patch_image.fold(pred_token_mask_unshuffle, (h, w))
    assert x_fold.shape == (b, c, h, w)
    assert pred_token_mask_fold.shape == (b, c, h, w)


def test_mae_forward_pass():
    mae = MAE()
    b, c, h, w = 8, 3, 64, 64
    n, d = 64, 192
    x = torch.randn(b, c, h, w)
    output_dict = mae(x)
    assert output_dict["pixel_preds"].shape == (b, n, d)
    assert output_dict["pred_token_mask"].shape == (b, n)


if __name__ == "__main__":
    test_patch_image()
    test_unshuffle_tokes()
    test_mae_forward_pass()
    test_patch_and_unshuffle()
    print("Test passed.")
