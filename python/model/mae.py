import torch
import torch.nn as nn


class PatchImage(nn.Module):
    def __init__(self, patch_kernel_size=8):
        super(PatchImage, self).__init__()
        self.patch_kernel_size = patch_kernel_size
        self.unfold = torch.nn.Unfold(
            kernel_size=patch_kernel_size, stride=patch_kernel_size
        )

    def forward(self, x):
        x = self.unfold(x)  # B x (C*P*P) x N
        x = x.permute(0, 2, 1)  # (B, N, C*P*P)
        return x

    def fold(self, x, output_size):
        x = x.permute(0, 2, 1)  # (B, C*P*P, N)
        fold = torch.nn.Fold(
            output_size=output_size,
            kernel_size=self.patch_kernel_size,
            stride=self.patch_kernel_size,
        )
        x = fold(x)
        return x


class EmbedPatches(nn.Module):
    def __init__(self, patch_size=192, embed_size=256, num_patches=64):
        super(EmbedPatches, self).__init__()
        self.embed_size = embed_size
        self.num_patches = num_patches
        self.patch_size = patch_size

        self.patch_embed_transform = nn.Linear(self.patch_size, self.embed_size)
        self.positional_embeddings = nn.Parameter(
            data=torch.randn(self.num_patches, embed_size), requires_grad=True
        )

    def forward(self, x):
        x_embed = self.patch_embed_transform(x)
        x_embed_pos = x_embed + self.positional_embeddings
        return x_embed_pos


class EmbedMasking(nn.Module):
    def __init__(self, mask_fraction=0.75):
        super(EmbedMasking, self).__init__()
        self.mask_fraction = mask_fraction

    def forward(self, x):
        b, n, embed_size = x.shape
        random_indices = torch.stack(
            [torch.randperm(n) for _ in range(b)], dim=0
        )  # (b, n)
        random_indices = random_indices.unsqueeze(-1).expand(
            -1,
            -1,
            embed_size,
        )
        random_indices = random_indices.to(x.device)
        x_shuffle = torch.gather(x, 1, random_indices)

        keep_size = int((1 - self.mask_fraction) * n)
        x_masked = x_shuffle[:, :keep_size, :]
        return x_masked, random_indices


class MAE(nn.Module):

    def __init__(self, patch_kernel_size=8, img_size=(3, 64, 64), embed_size=256, mask_fraction=0.75):
        super(MAE, self).__init__()
        self.patch_kernel_size = patch_kernel_size
        self.img_size = img_size
        self.num_patches = int((img_size[1] / patch_kernel_size) ** 2)
        self.patch_dim = int((img_size[1] / patch_kernel_size) ** 2 * img_size[0])
        self.embed_size = embed_size
        self.mask_fraction = mask_fraction

        self.patch_image = PatchImage(patch_kernel_size=self.patch_kernel_size)
        self.embed_patches = EmbedPatches(
            patch_size=self.patch_dim, embed_size=self.embed_size
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size, nhead=4
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=2
        )
        self.embed_mask = EmbedMasking(mask_fraction=self.mask_fraction)
        self.mask_tokens = nn.Parameter(
            data=torch.randn(
                int(self.num_patches * (self.mask_fraction)), self.embed_size
            ),
            requires_grad=True,
        )
        self.embed_patches_for_decoder = EmbedPatches(
            patch_size=self.embed_size, embed_size=self.embed_size
        )
        self.transformer_decoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=1
        )

        self.pixel_prediction = nn.Linear(self.embed_size, self.patch_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_patch = self.patch_image(x)
        x_embed = self.embed_patches(x_patch)
        x_masked, random_indices = self.embed_mask(x_embed)
        x_encoder = self.transformer_encoder(x_masked)

        mask_tokens = self.mask_tokens.unsqueeze(0).expand(b, -1, -1)
        x_encoder_w_masked_tokens = torch.cat([x_encoder, mask_tokens], 1)
        pred_token_mask = torch.ones(
            b, self.num_patches, (self.patch_kernel_size**2) * c, device=x.device
        )
        pred_token_mask[:, : x_encoder.shape[1], :] = 0

        x_unshuffle = self.unshuffle_tokens(x_encoder_w_masked_tokens, random_indices)
        x_unshuffle_embed = self.embed_patches_for_decoder(x_unshuffle)
        x_decode = self.transformer_decoder(x_unshuffle_embed)

        pixel_preds = self.pixel_prediction(x_decode)
        pixel_preds = self.patch_image.fold(pixel_preds, output_size=(h, w))
        pred_token_mask = self.patch_image.fold(pred_token_mask, output_size=(h, w))

        return_dict = {
            "pixel_preds": pixel_preds,
            "pred_token_mask": pred_token_mask.bool(),
        }
        return return_dict

    def unshuffle_tokens(self, x_encoder_w_masked_tokens, random_indices):
        b, n, d = random_indices.shape
        perm = random_indices[:, :, 0]
        inv = torch.empty_like(perm, device=x_encoder_w_masked_tokens.device)
        for i in range(b):
            inv[i, perm[i]] = torch.arange(n, device=x_encoder_w_masked_tokens.device)
        inv = inv.unsqueeze(-1).expand(-1, -1, d)
        x_unshuffle = torch.gather(x_encoder_w_masked_tokens, 1, inv)
        return x_unshuffle


if __name__ == "__main__":
    model = MAE()
    x = torch.randn(8, 3, 64, 64)
    out = model(x)
