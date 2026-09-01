"""Aggregate LODO fold results into a summary table.

Reads checkpoints/<run_name>/results.json for all completed folds and prints
per-variant tables (ResNet18-asymmetric, MAE fine-tune, MAE linear probe) plus
a cross-variant summary comparing Track 1 vs Track 2.

Usage:
    python scripts/aggregate_lodo_results.py
    python scripts/aggregate_lodo_results.py --run-prefix resnet18-asymmetric  # single variant
"""

import argparse
import json
from pathlib import Path

import numpy as np

DETECTOR_ORDER = {1: "AGIPD", 2: "JUNGFRAU_4M", 3: "ePix10k", 4: "Eiger4M"}

# All run-name prefixes the script knows about, in display order.
ALL_PREFIXES = [
    ("resnet18-asymmetric", "ResNet18-asymmetric (Track 1)"),
    ("vits16-mae-finetune", "ViT-S/16 MAE fine-tune (Track 2)"),
    ("vits16-mae-probe", "ViT-S/16 MAE linear probe (Track 2)"),
]


def _variant_label(data: dict, prefix: str) -> str:
    """Derive a display label from results.json keys when present."""
    track = data.get("track")
    probe = data.get("probe")
    if track == "ssl":
        return "MAE-probe" if probe else "MAE-finetune"
    return prefix


def load_results(checkpoints_dir: Path, run_prefix: str) -> list[dict]:
    results = []
    for path in sorted(checkpoints_dir.glob(f"{run_prefix}-fold*/results.json")):
        with open(path) as f:
            data = json.load(f)
        fold_id = data.get("fold_id")
        if fold_id not in DETECTOR_ORDER:
            continue
        cross = data.get("cross", {})
        if any(
            cross.get(k, float("nan")) != cross.get(k, float("nan"))
            for k in ("ap", "auc_roc", "f1")
        ):
            print(f"  Warning: fold {fold_id} has NaN metrics — skipping.")
            continue
        results.append(data)
    return sorted(results, key=lambda r: r["fold_id"])


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


_HDR = f"{'Fold':<6} {'Held-out':<14} {'Cross AP':>9} {'Cross AUC':>10} {'Cross F1':>9} {'ID AP':>8} {'ID AUC':>9} {'ID F1':>8}"
_SEP = "-" * len(_HDR)


def print_variant_table(label: str, results: list[dict]) -> tuple[float, float]:
    """Print one variant table and return (mean_cross_ap, std_cross_ap)."""
    print(f"\n## {label}")
    print(_HDR)
    print(_SEP)
    cross_aps, cross_aucs, cross_f1s = [], [], []
    for r in results:
        fold = r["fold_id"]
        det = r["test_detector"]
        c = r["cross"]
        d = r["in_domain"]
        print(
            f"{fold:<6} {det:<14} {c['ap']:>9.4f} {c['auc_roc']:>10.4f} {c['f1']:>9.4f}"
            f" {d['ap']:>8.4f} {d['auc_roc']:>9.4f} {d['f1']:>8.4f}"
        )
        cross_aps.append(c["ap"])
        cross_aucs.append(c["auc_roc"])
        cross_f1s.append(c["f1"])
    print(_SEP)
    ap_mu, ap_sd = mean_std(cross_aps)
    auc_mu, auc_sd = mean_std(cross_aucs)
    f1_mu, f1_sd = mean_std(cross_f1s)
    n = len(results)
    print(
        f"{'Mean':<6} {f'({n}/4 folds)':<14} {ap_mu:>9.4f} {auc_mu:>10.4f} {f1_mu:>9.4f}"
    )
    if n > 1:
        print(f"{'Std':<6} {'':<14} {ap_sd:>9.4f} {auc_sd:>10.4f} {f1_sd:>9.4f}")
    return ap_mu, ap_sd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Checkpoint directory prefix to aggregate (single-variant mode). "
            "Omit to show all known variants."
        ),
    )
    args = parser.parse_args()

    if args.run_prefix is not None:
        # Single-variant mode: original behavior.
        print(
            f"Aggregating: {args.checkpoints_dir}/{args.run_prefix}-fold*/results.json\n"
        )
        results = load_results(args.checkpoints_dir, args.run_prefix)
        if not results:
            print("No completed fold results found.")
            return
        print_variant_table(args.run_prefix, results)
        return

    # Multi-variant mode: scan all known prefixes.
    summary_rows: list[tuple[str, float, float]] = []
    any_found = False
    for prefix, label in ALL_PREFIXES:
        results = load_results(args.checkpoints_dir, prefix)
        if not results:
            continue
        any_found = True
        ap_mu, ap_sd = print_variant_table(label, results)
        summary_rows.append((label, ap_mu, ap_sd))

    if not any_found:
        print("No completed fold results found in any known variant.")
        return

    if len(summary_rows) > 1:
        print("\n## Cross-variant summary (mean cross AP)")
        print(f"{'Variant':<40} {'Mean AP':>8} {'Std AP':>8}")
        print("-" * 58)
        for label, mu, sd in summary_rows:
            sd_str = f"{sd:>8.4f}" if not (sd != sd) else f"{'—':>8}"
            print(f"{label:<40} {mu:>8.4f} {sd_str}")


if __name__ == "__main__":
    main()
