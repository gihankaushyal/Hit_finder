"""Plot realized train/realized_hit_frac vs epoch across all folds of a LODO sweep.

Usage:
    python scripts/plot_hit_frac.py --name-prefix resnet18-asymmetric-seed42 \
        --project sfx-hitfinder [--entity ENTITY] \
        [--out docs/figures/hit_frac_diagnostics/resnet18-asymmetric-seed42.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


def main(name_prefix: str, project: str, entity: str | None, out: Path) -> None:
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = [r for r in api.runs(path) if r.name.startswith(f"{name_prefix}-fold")]
    if not runs:
        raise SystemExit(
            f"No runs found with name prefix '{name_prefix}-fold' in {path}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    configured_hit_frac = None
    for run in sorted(runs, key=lambda r: r.name):
        hist = run.history(keys=["epoch", "train/realized_hit_frac"])
        if hist.empty:
            continue
        label = f"{run.name} ({run.config.get('test_detector', '?')})"
        ax.plot(hist["epoch"], hist["train/realized_hit_frac"], marker=".", label=label)
        configured_hit_frac = run.config.get("asymmetric", {}).get(
            "hit_frac", configured_hit_frac
        )

    if configured_hit_frac is not None:
        ax.axhline(
            configured_hit_frac,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"configured hit_frac={configured_hit_frac}",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("realized train hit fraction")
    ax.set_title(f"Realized hit fraction vs epoch — {name_prefix}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = (
        args.out
        or Path("docs/figures/hit_frac_diagnostics") / f"{args.name_prefix}.png"
    )
    main(args.name_prefix, args.project, args.entity, out)
