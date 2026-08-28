"""Aggregate LODO fold results into a summary table.

Reads checkpoints/<run_name>/results.json for all asymmetric folds and
prints per-fold metrics plus mean ± std across completed folds.

Usage:
    python scripts/aggregate_lodo_results.py
    python scripts/aggregate_lodo_results.py --run-prefix resnet18-lodo   # old baseline
    python scripts/aggregate_lodo_results.py --run-prefix resnet18-asymmetric  # new pipeline (default)
"""

import argparse
import json
from pathlib import Path

import numpy as np

DETECTOR_ORDER = {1: "AGIPD", 2: "JUNGFRAU_4M", 3: "ePix10k", 4: "Eiger4M"}


def load_results(checkpoints_dir: Path, run_prefix: str) -> list[dict]:
    results = []
    for path in sorted(checkpoints_dir.glob(f"{run_prefix}-fold*/results.json")):
        with open(path) as f:
            data = json.load(f)
        fold_id = data.get("fold_id")
        if fold_id not in DETECTOR_ORDER:
            continue
        cross = data.get("cross", {})
        _metric_keys = {"ap", "auc_roc", "f1"}
        if any(
            cross.get(k, float("nan")) != cross.get(k, float("nan"))
            for k in _metric_keys
        ):
            print(f"  Warning: fold {fold_id} has NaN metrics — skipping.")
            continue
        results.append(data)
    return sorted(results, key=lambda r: r["fold_id"])


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-prefix",
        default="resnet18-asymmetric",
        help="Checkpoint directory prefix to aggregate (default: resnet18-asymmetric)",
    )
    args = parser.parse_args()

    print(f"Aggregating: {args.checkpoints_dir}/{args.run_prefix}-fold*/results.json\n")
    results = load_results(args.checkpoints_dir, args.run_prefix)

    if not results:
        print("No completed fold results found.")
        return

    header = f"{'Fold':<6} {'Held-out':<14} {'Cross AP':>9} {'Cross AUC':>10} {'Cross F1':>9} {'ID AP':>8} {'ID AUC':>9} {'ID F1':>8}"
    print(header)
    print("-" * len(header))

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

    print("-" * len(header))
    ap_mu, ap_sd = mean_std(cross_aps)
    auc_mu, auc_sd = mean_std(cross_aucs)
    f1_mu, f1_sd = mean_std(cross_f1s)
    n = len(results)
    print(
        f"{'Mean':<6} {f'({n}/4 folds)':<14} {ap_mu:>9.4f} {auc_mu:>10.4f} {f1_mu:>9.4f}"
    )
    if n > 1:
        print(f"{'Std':<6} {'':<14} {ap_sd:>9.4f} {auc_sd:>10.4f} {f1_sd:>9.4f}")


if __name__ == "__main__":
    main()
