"""Aggregate per-fold results.json files into a summary table.

Run after all LODO folds have completed (sequentially or in parallel):

    python scripts/aggregate_lodo_results.py
    python scripts/aggregate_lodo_results.py --checkpoints checkpoints/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import format_results_table


def main(checkpoints_dir: Path) -> None:
    result_files = sorted(checkpoints_dir.glob("*-lodo-fold*-seed*/results.json"))

    if not result_files:
        print(f"No results.json files found under {checkpoints_dir}")
        return

    fold_results: dict[str, dict] = {}
    ap_values: list[float] = []

    for path in result_files:
        with open(path) as f:
            r = json.load(f)

        key = f"fold_{r['fold_id']}"
        fold_results[key] = {
            "test_detector": r["test_detector"],
            "ap":            r["cross"]["ap"],
            "auc_roc":       r["cross"]["auc_roc"],
            "f1":            r["cross"]["f1"],
            "in_domain_ap":  r["in_domain"]["ap"],
        }
        ap_values.append(r["cross"]["ap"])
        print(
            f"  {key}  ({r['test_detector']:12s})  "
            f"cross AP={r['cross']['ap']:.4f}  "
            f"AUC={r['cross']['auc_roc']:.4f}  "
            f"F1={r['cross']['f1']:.4f}  "
            f"in-domain AP={r['in_domain']['ap']:.4f}"
        )

    results_for_table: dict = {k: {"ap": v["ap"], "test_detector": v["test_detector"]}
                                for k, v in fold_results.items()}

    if len(ap_values) > 1:
        results_for_table["mean_ap"] = float(np.mean(ap_values))
        results_for_table["std_ap"] = float(np.std(ap_values, ddof=1))
    elif ap_values:
        results_for_table["mean_ap"] = ap_values[0]
        results_for_table["std_ap"] = float("nan")

    print("\n" + format_results_table(results_for_table))

    missing = {1, 2, 3, 4} - {json.load(open(p))["fold_id"] for p in result_files}
    if missing:
        print(f"  Note: folds {sorted(missing)} not yet complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate LODO fold results")
    parser.add_argument(
        "--checkpoints",
        default="checkpoints",
        help="Root checkpoints directory (default: checkpoints/)",
    )
    args = parser.parse_args()
    main(Path(args.checkpoints))
