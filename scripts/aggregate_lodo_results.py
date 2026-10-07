"""Aggregate LODO fold results into a summary table.

Reads checkpoints/<run_name>/results.json for completed folds and prints
per-variant tables plus a cross-variant summary. Run-name prefixes are
discovered from what's actually on disk under --checkpoints-dir — no
hardcoded list to maintain. A small set of doc-only historical generations
(baselines recorded in docs/progress_notes.md whose checkpoints have since
been overwritten or never existed on disk) is also offered for comparison.

Usage:
    python scripts/aggregate_lodo_results.py                  # interactive menu
    python scripts/aggregate_lodo_results.py --select all      # every discovered + doc variant
    python scripts/aggregate_lodo_results.py --select 1,3,4    # specific menu entries, no prompt
    python scripts/aggregate_lodo_results.py --run-prefix resnet18-asymmetric-v2  # single variant, legacy mode
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

DETECTOR_ORDER = {1: "AGIPD", 2: "JUNGFRAU_4M", 3: "ePix10k", 4: "Eiger4M"}

_RUN_DIR_RE = re.compile(r"^(?P<prefix>.+)-fold\d+-seed\d+$")

# Doc-only generations that predate this checkpoint format or have since been
# overwritten on disk. Source: docs/progress_notes.md. These never come from
# results.json — value is (label, mean_cross_ap, std_cross_ap, source).
DOC_BASELINES = {
    "naive-baseline": (
        "Naive baseline (§6, 2026-06-27)",
        0.812,
        0.167,
        "docs/progress_notes.md:112",
    ),
    "asymmetric-v1": (
        "Asymmetric pipeline v1 (§14, 2026-08-24, pre-frame-cache)",
        0.839,
        0.021,
        "docs/progress_notes.md:485",
    ),
}


def discover_prefixes(checkpoints_dir: Path) -> list[str]:
    """Find run-name prefixes with at least one completed fold on disk."""
    prefixes = set()
    if not checkpoints_dir.is_dir():
        return []
    for p in checkpoints_dir.iterdir():
        if not p.is_dir():
            continue
        m = _RUN_DIR_RE.match(p.name)
        if m and (p / "results.json").exists():
            prefixes.add(m.group("prefix"))
    return sorted(prefixes)


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


def print_doc_baseline_row(key: str) -> tuple[float, float]:
    """Print a one-line synthetic row for a doc-only baseline (no per-fold data)."""
    label, mu, sd, source = DOC_BASELINES[key]
    print(f"\n## {label}  (doc-only — source: {source})")
    print(f"  Mean cross AP: {mu:.4f}  +/- {sd:.4f}  (no per-fold breakdown available)")
    return mu, sd


def build_menu(
    checkpoints_dir: Path,
) -> list[tuple[str, str, bool]]:
    """Return numbered menu entries as (key, display_label, is_live)."""
    entries = []
    for prefix in discover_prefixes(checkpoints_dir):
        entries.append((prefix, f"{prefix}  (live, {checkpoints_dir}/)", True))
    for key, (label, *_rest) in DOC_BASELINES.items():
        entries.append((key, f"{label}  (doc)", False))
    return entries


def prompt_selection(menu: list[tuple[str, str, bool]]) -> list[int]:
    print("Available results:")
    for i, (_key, label, _is_live) in enumerate(menu, start=1):
        print(f"  {i}. {label}")
    while True:
        choice = input(
            "Select variant(s) to aggregate — comma-separated numbers, or 'all': "
        ).strip()
        if choice.lower() == "all":
            return list(range(1, len(menu) + 1))
        try:
            indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
        except ValueError:
            print("Please enter comma-separated numbers (e.g. 1,3,4) or 'all'.")
            continue
        if all(1 <= i <= len(menu) for i in indices) and indices:
            return indices
        print(f"Numbers must be between 1 and {len(menu)}.")


def parse_select_flag(select: str, menu_len: int) -> list[int]:
    if select.lower() == "all":
        return list(range(1, menu_len + 1))
    try:
        indices = [int(x.strip()) for x in select.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(
            f"Invalid --select value: {select!r}. Use e.g. '1,3,4' or 'all'."
        )
    if not indices or not all(1 <= i <= menu_len for i in indices):
        raise SystemExit(f"--select indices must be between 1 and {menu_len}.")
    return indices


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Checkpoint directory prefix to aggregate (single-variant mode, no "
            "menu). Omit to use the interactive menu or --select."
        ),
    )
    parser.add_argument(
        "--select",
        default=None,
        help=(
            "Non-interactive escape hatch: comma-separated menu numbers (e.g. "
            "'1,3,4') or 'all'. Skips the interactive prompt."
        ),
    )
    args = parser.parse_args()

    if args.run_prefix is not None:
        # Single-variant mode: original behavior, bypasses the menu entirely.
        print(
            f"Aggregating: {args.checkpoints_dir}/{args.run_prefix}-fold*/results.json\n"
        )
        results = load_results(args.checkpoints_dir, args.run_prefix)
        if not results:
            print("No completed fold results found.")
            return
        print_variant_table(args.run_prefix, results)
        return

    menu = build_menu(args.checkpoints_dir)
    if not menu:
        print("No live run results or doc baselines available.")
        return

    if args.select is not None:
        indices = parse_select_flag(args.select, len(menu))
    else:
        indices = prompt_selection(menu)

    summary_rows: list[tuple[str, float, float]] = []
    for i in indices:
        key, label, is_live = menu[i - 1]
        if is_live:
            results = load_results(args.checkpoints_dir, key)
            if not results:
                print(f"\n## {label}\n  No completed fold results found — skipping.")
                continue
            ap_mu, ap_sd = print_variant_table(key, results)
        else:
            ap_mu, ap_sd = print_doc_baseline_row(key)
        summary_rows.append((label, ap_mu, ap_sd))

    if len(summary_rows) > 1:
        print("\n## Cross-variant summary (mean cross AP)")
        print(f"{'Variant':<55} {'Mean AP':>8} {'Std AP':>8}")
        print("-" * 73)
        for label, mu, sd in summary_rows:
            sd_str = f"{sd:>8.4f}" if not (sd != sd) else f"{'—':>8}"
            print(f"{label:<55} {mu:>8.4f} {sd_str}")


if __name__ == "__main__":
    main()
