"""Visualize MAE reconstructions: masked input | reconstruction | target.

Usage:
    python scripts/visualize_mae_recon.py \
        --checkpoint checkpoints/mae-vits16-fold1-seed42/last.pt \
        --config configs/ssl/mae_pretrain.yaml --fold 1 \
        --n-samples 8 --out docs/figures/mae_recon/fold1.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.dataloader import ssl_crop_loader
from src.evaluation.benchmark import build_lodo_folds
from src.models.ssl import build_mae_model
from src.training.lodo import build_sessions
from src.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    model = build_mae_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    sessions, session_map = build_sessions(cfg["lodo"])
    fold = next(f for f in build_lodo_folds() if f["fold_id"] == args.fold)
    ids = [s["session_id"] for s in sessions if s["detector"] != fold["test_detector"]]
    dl = ssl_crop_loader(
        session_map,
        ids,
        batch_size=args.n_samples,
        num_workers=0,
        shuffle=True,
        seed=cfg["seed"],
    )
    crops, _ = next(iter(dl))

    with torch.no_grad():
        _, pred, mask = model(crops, mask_ratio=cfg["ssl"]["mask_ratio"])
    recon = model.unpatchify(pred)
    mask_img = model.unpatchify(
        mask.unsqueeze(-1).expand(-1, -1, model.patch_size**2 * model.in_chans)
    )
    masked_input = crops * (1 - mask_img)
    composite = crops * (1 - mask_img) + recon * mask_img

    n = crops.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    titles = ["masked input", "reconstruction", "recon+visible", "target"]
    panels = [masked_input, recon, composite, crops]
    for i in range(n):
        for j in range(4):
            ax = axes[i][j] if n > 1 else axes[j]
            ax.imshow(panels[j][i, 0].numpy(), cmap="viridis")
            ax.set_axis_off()
            if i == 0:
                ax.set_title(titles[j])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
