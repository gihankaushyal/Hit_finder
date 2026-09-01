"""MAE ViT encoder + classification head. Track 2 — self-supervised. NOT ResNet.

Design: docs/superpowers/specs/2026-09-01-phase5-ssl-mae-design.md
Masking pipelines (YAML `ssl.masking`): "random" (default, ratio 0.6 — risk B3)
and "peak_aware" (hitfinder-centroid-biased).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MAE_MASK_RATIO_DEFAULT: float = 0.6
PEAK_MASK_FRAC_DEFAULT: float = 1.0
# Bias added to shuffle noise so selected peak patches sort into the masked tail.
_PEAK_NOISE_BIAS: float = 2.0


def _masking_from_noise(
    noise: torch.Tensor, mask_ratio: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn per-token noise into (ids_keep, mask, ids_restore).

    Tokens with the SMALLEST noise are kept (MAE convention).
    mask is (B, L) with 1 = masked, 0 = kept.
    """
    B, L = noise.shape
    len_keep = int(L * (1 - mask_ratio))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones(B, L, device=noise.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return ids_keep, mask, ids_restore


def random_masking(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = MAE_MASK_RATIO_DEFAULT,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAE-standard per-sample random masking."""
    noise = torch.rand(batch_size, seq_len, device=device, generator=generator)
    return _masking_from_noise(noise, mask_ratio)


def peak_aware_masking(
    peak_patches: torch.Tensor,
    mask_ratio: float = MAE_MASK_RATIO_DEFAULT,
    peak_mask_frac: float = PEAK_MASK_FRAC_DEFAULT,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random masking biased so hitfinder-peak patches land in the masked set.

    peak_patches: (B, L) bool — True where the token's patch contains a
    hitfinder centroid. Per sample, round(peak_mask_frac * n_peaks) peak
    patches are force-masked (capped at the mask budget); remaining budget is
    filled randomly. Samples with no peaks degrade to plain random masking.
    """
    B, L = peak_patches.shape
    device = peak_patches.device
    noise = torch.rand(B, L, device=device, generator=generator)
    budget = L - int(L * (1 - mask_ratio))
    for b in range(B):
        peak_idx = torch.nonzero(peak_patches[b], as_tuple=False).flatten()
        n_force = min(int(round(peak_mask_frac * len(peak_idx))), budget)
        if n_force == 0:
            continue
        perm = torch.randperm(len(peak_idx), generator=generator)[:n_force]
        noise[b, peak_idx[perm]] += _PEAK_NOISE_BIAS
    return _masking_from_noise(noise, mask_ratio)


from timm.models.vision_transformer import Block, PatchEmbed  # noqa: E402

VIT_SMALL = {"embed_dim": 384, "depth": 12, "num_heads": 6}
DECODER_DEFAULTS = {
    "decoder_embed_dim": 256,
    "decoder_depth": 4,
    "decoder_num_heads": 8,
}
IMG_SIZE_DEFAULT = 224
PATCH_SIZE_DEFAULT = 16
MASKING_RANDOM = "random"
MASKING_PEAK_AWARE = "peak_aware"


def _sincos_pos_embed_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / 10000 ** (omega / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> np.ndarray:
    """Fixed 2D sine-cosine positional embedding (MAE convention)."""
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _sincos_pos_embed_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_pos_embed_1d(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MAEViT(nn.Module):
    """Masked autoencoder with a ViT encoder and lightweight decoder."""

    def __init__(
        self,
        img_size: int = IMG_SIZE_DEFAULT,
        patch_size: int = PATCH_SIZE_DEFAULT,
        in_chans: int = 1,
        embed_dim: int = VIT_SMALL["embed_dim"],
        depth: int = VIT_SMALL["depth"],
        num_heads: int = VIT_SMALL["num_heads"],
        decoder_embed_dim: int = DECODER_DEFAULTS["decoder_embed_dim"],
        decoder_depth: int = DECODER_DEFAULTS["decoder_depth"],
        decoder_num_heads: int = DECODER_DEFAULTS["decoder_num_heads"],
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_chans, bias=True
        )
        self._init_weights()

    def _init_weights(self) -> None:
        grid = int(self.patch_embed.num_patches**0.5)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        dpe = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dpe).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p, c = self.patch_size, self.in_chans
        B, _, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, c, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(B, h * w, p * p * c)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, c = self.patch_size, self.in_chans
        B, L, _ = x.shape
        h = w = int(L**0.5)
        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(B, c, h * p, w * p)

    def forward_encoder(
        self, imgs: torch.Tensor, ids_keep: torch.Tensor
    ) -> torch.Tensor:
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        x = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_decoder(
        self, latent: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_embed(latent)
        B, L = ids_restore.shape
        n_mask = L + 1 - x.shape[1]
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)[:, 1:, :]  # drop cls

    def reconstruction_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        err = (pred - target) ** 2
        if valid_mask is not None:
            pix_w = self.patchify(valid_mask)
            per_patch = (err * pix_w).sum(dim=-1) / pix_w.sum(dim=-1).clamp(min=1.0)
        else:
            per_patch = err.mean(dim=-1)
        return (per_patch * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = MAE_MASK_RATIO_DEFAULT,
        masking: str = MASKING_RANDOM,
        peak_patches: torch.Tensor | None = None,
        peak_mask_frac: float = PEAK_MASK_FRAC_DEFAULT,
        valid_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = imgs.shape[0]
        L = self.patch_embed.num_patches
        if masking == MASKING_RANDOM:
            ids_keep, mask, ids_restore = random_masking(
                B, L, mask_ratio, device=imgs.device, generator=generator
            )
        elif masking == MASKING_PEAK_AWARE:
            if peak_patches is None:
                peak_patches = torch.zeros(B, L, dtype=torch.bool, device=imgs.device)
            ids_keep, mask, ids_restore = peak_aware_masking(
                peak_patches, mask_ratio, peak_mask_frac, generator=generator
            )
        else:
            raise ValueError(f"Unknown masking mode: {masking!r}")
        latent = self.forward_encoder(imgs, ids_keep)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.reconstruction_loss(imgs, pred, mask, valid_mask=valid_mask)
        return loss, pred, mask


def build_mae_model(cfg: dict) -> MAEViT:
    """Build the MAE from the `ssl:` config block (ViT-S/16 defaults)."""
    ssl_cfg = cfg.get("ssl", {})
    return MAEViT(
        embed_dim=ssl_cfg.get("embed_dim", VIT_SMALL["embed_dim"]),
        depth=ssl_cfg.get("depth", VIT_SMALL["depth"]),
        num_heads=ssl_cfg.get("num_heads", VIT_SMALL["num_heads"]),
        decoder_embed_dim=ssl_cfg.get(
            "decoder_embed_dim", DECODER_DEFAULTS["decoder_embed_dim"]
        ),
        decoder_depth=ssl_cfg.get("decoder_depth", DECODER_DEFAULTS["decoder_depth"]),
        decoder_num_heads=ssl_cfg.get(
            "decoder_num_heads", DECODER_DEFAULTS["decoder_num_heads"]
        ),
        norm_pix_loss=ssl_cfg.get("norm_pix_loss", False),
    )
