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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metric_keys import RECALL_FPR_KEYS  # noqa: E402
from src.utils.config import DEFAULT_SEED as CONFIG_DEFAULT_SEED  # noqa: E402

DETECTOR_ORDER = {1: "AGIPD", 2: "JUNGFRAU_4M", 3: "ePix10k", 4: "Eiger4M"}
# The side file an --inference-only pass writes next to a finished run's results.json
# (src/training/inference_results.py; not imported, to keep this script free of torch).
INFERENCE_RESULTS = "results.inference.json"
_SPLITS = ("cross", "in_domain")
# Keys --metric accepts: the per-split keys of a result file
# (SPLIT_RESULT_KEYS in src/training/inference_results.py).
METRIC_CHOICES = ("ap", "auc_roc", "f1", "threshold", *RECALL_FPR_KEYS)
# Set on a loaded record: the splits whose --metric value came from INFERENCE_RESULTS.
FROM_INFERENCE = "_metric_from_inference"

# A run directory is "<prefix>-fold<N>-seed<S><suffix>". The optional suffix (e.g. "-v2" from
# wandb.run_suffix, or "-careful" / "-lpft" from --run-suffix) is part of the group key, so a
# recipe experiment never mixes into the unsuffixed baseline of the same prefix.
_RUN_DIR_RE = re.compile(r"^(?P<prefix>.+)-fold\d+-seed(?P<seed>\d+)(?P<suffix>.*)$")
_GROUP_SEP = "-fold*-seed"
# The seed the configs default to. A seed-42 unsuffixed run is keyed by its bare prefix; every
# other group names its seed ("<prefix>-fold*-seed43<suffix>"), because seeds of one prefix are
# repeats of the same experiment and must never be averaged into one fold row. Pooling every seed
# is an explicit opt-in (--pool-seeds); a legacy "seed*" key keeps its old meaning, seed 42 only.
DEFAULT_SEED = str(CONFIG_DEFAULT_SEED)
_KEY_RE = re.compile(
    r"^(?P<prefix>.+?)(?:"
    + re.escape(_GROUP_SEP)
    + r"(?P<seed>\d+|\*)(?P<suffix>.*))?$"
)

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


def discover_prefixes(checkpoints_dir: Path, pool_seeds: bool = False) -> list[str]:
    """Find run groups with at least one completed fold on disk.

    A group key is the bare prefix for unsuffixed seed-42 runs and
    "<prefix>-fold*-seed<S><suffix>" for everything else, so suffixed runs and other seeds
    never share a key with the baseline. With pool_seeds every seed of a prefix and suffix
    collapses into one "<prefix>-fold*-seed*<suffix>" group, so the menu lists it once.
    """
    groups: set[str] = set()
    if not checkpoints_dir.is_dir():
        return []
    for p in checkpoints_dir.iterdir():
        if not p.is_dir():
            continue
        m = _RUN_DIR_RE.match(p.name)
        if m and (p / "results.json").exists():
            suffix = m.group("suffix")
            seed = m.group("seed")
            if pool_seeds:
                groups.add(m.group("prefix") + _GROUP_SEP + "*" + suffix)
            elif suffix or seed != DEFAULT_SEED:
                groups.add(m.group("prefix") + _GROUP_SEP + seed + suffix)
            else:
                groups.add(m.group("prefix"))
    return sorted(groups)


def load_results(
    checkpoints_dir: Path,
    run_prefix: str,
    pool_seeds: bool = False,
    metric: str | None = None,
) -> list[dict]:
    """Load results.json for every fold of one run group (see discover_prefixes).

    pool_seeds=True matches every seed of the group's prefix and suffix; otherwise a key names
    exactly one seed (a legacy "seed*" key means the default seed, as it did before seeds existed).
    metric: a split key that results.json may lack (a run evaluated before the key existed);
    it is then taken from results.inference.json, and only that key.
    """
    key = _KEY_RE.match(run_prefix)
    prefix = key.group("prefix")
    seed = key.group("seed")
    if seed is None or seed == "*":
        seed = DEFAULT_SEED
    seed_re = r"\d+" if pool_seeds else re.escape(seed)
    suffix = key.group("suffix") or ""
    dir_re = re.compile(
        re.escape(prefix) + r"-fold\d+-seed" + seed_re + re.escape(suffix)
    )
    results = []
    for path in sorted(checkpoints_dir.glob("*/results.json")):
        if not dir_re.fullmatch(path.parent.name):
            continue
        with open(path) as f:
            data = json.load(f)
        if metric is not None:
            _fill_metric_from_inference(data, path.with_name(INFERENCE_RESULTS), metric)
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
    if pool_seeds:
        results = _pool_by_fold(results)
    return sorted(results, key=lambda r: r["fold_id"])


def _fill_metric_from_inference(data: dict, side: Path, metric: str) -> None:
    """Copy `metric` into data's splits from the inference side file where data lacks it."""
    missing = [s for s in _SPLITS if metric not in data.get(s, {})]
    if not missing or not side.exists():
        return
    extra = json.loads(side.read_text())
    for s in missing:
        if metric in extra.get(s, {}):
            data.setdefault(s, {})[metric] = extra[s][metric]
            data.setdefault(FROM_INFERENCE, []).append(s)


def _pool_by_fold(results: list[dict]) -> list[dict]:
    """Average the per-seed results of each fold into one record (n_seeds notes how many)."""
    by_fold: dict[int, list[dict]] = {}
    for r in results:
        by_fold.setdefault(r["fold_id"], []).append(r)
    pooled = []
    for runs in by_fold.values():
        rec = {
            "fold_id": runs[0]["fold_id"],
            "test_detector": runs[0]["test_detector"],
            "n_seeds": len(runs),
        }
        filled = sorted({s for r in runs for s in r.get(FROM_INFERENCE, [])})
        if filled:
            rec[FROM_INFERENCE] = filled
        for split in ("cross", "in_domain"):
            rec[split] = {
                k: float(np.mean([r[split][k] for r in runs]))
                for k, v in runs[0].get(split, {}).items()
                if isinstance(v, (int, float))
                and all(isinstance(r[split].get(k), (int, float)) for r in runs)
            }
        pooled.append(rec)
    return pooled


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


def print_metric_table(
    label: str, results: list[dict], metric: str
) -> tuple[float, float]:
    """One variant's table for a single metric (cross and in-domain); n/a where missing.

    A value read from results.inference.json is marked with "*" (for pooled seeds: at
    least one seed's value). Returns (mean, std) of the cross value over the folds that
    have it.
    """
    print(f"\n## {label}  ({metric})")
    print(f"{'Fold':<6} {'Held-out':<14} {'Cross':>9} {'ID':>9}")
    print("-" * 41)

    def fmt(v: float) -> str:
        return f"{v:>9.4f}" if v == v else f"{'n/a':>9}"

    cross_vals = []
    any_filled = False
    for r in results:
        filled = r.get(FROM_INFERENCE, [])
        any_filled = any_filled or bool(filled)
        c = r["cross"].get(metric, float("nan"))
        d = r["in_domain"].get(metric, float("nan"))
        mc = "*" if "cross" in filled else " "
        md = "*" if "in_domain" in filled else " "
        print(f"{r['fold_id']:<6} {r['test_detector']:<14} {fmt(c)}{mc}{fmt(d)}{md}")
        if c == c:
            cross_vals.append(c)
    mu, sd = mean_std(cross_vals)
    print(f"{'Mean':<6} {f'({len(cross_vals)}/4 folds)':<14} {fmt(mu)}")
    if len(cross_vals) > 1:
        print(f"{'Std':<6} {'':<14} {fmt(sd)}")
    if any_filled:
        print(
            f"* from {INFERENCE_RESULTS} (an --inference-only pass; see its"
            " 'inference' block for the evaluation settings)"
        )
    return mu, sd


def print_doc_baseline_row(key: str) -> tuple[float, float]:
    """Print a one-line synthetic row for a doc-only baseline (no per-fold data)."""
    label, mu, sd, source = DOC_BASELINES[key]
    print(f"\n## {label}  (doc-only — source: {source})")
    print(f"  Mean cross AP: {mu:.4f}  +/- {sd:.4f}  (no per-fold breakdown available)")
    return mu, sd


def build_menu(
    checkpoints_dir: Path, pool_seeds: bool = False
) -> list[tuple[str, str, bool]]:
    """Return numbered menu entries as (key, display_label, is_live)."""
    entries = []
    for prefix in discover_prefixes(checkpoints_dir, pool_seeds):
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


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Checkpoint directory prefix to aggregate (single-variant mode, no "
            "menu); unsuffixed seed-42 runs only. For any other group pass the key the "
            "menu prints, '<prefix>-fold*-seed<S><suffix>'. "
            "Omit to use the interactive menu or --select."
        ),
    )
    parser.add_argument(
        "--pool-seeds",
        action="store_true",
        help=(
            "Average every seed of each selected group into one fold row. Off by default: "
            "seeds are repeats of one experiment and are reported separately."
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
    parser.add_argument(
        "--metric",
        default=None,
        choices=METRIC_CHOICES,
        help=(
            "Tabulate one key of the cross / in_domain blocks (for example "
            "recall_at_fpr_1e-3) instead of AP/AUC/F1. Read from results.inference.json "
            "when results.json lacks it (marked *); n/a where neither has it."
        ),
    )
    args = parser.parse_args(argv)

    def print_table(label: str, results: list[dict]) -> tuple[float, float]:
        if args.metric is None:
            return print_variant_table(label, results)
        return print_metric_table(label, results, args.metric)

    if args.run_prefix is not None:
        # Single-variant mode: original behavior, bypasses the menu entirely.
        print(
            f"Aggregating: {args.checkpoints_dir}/{args.run_prefix}"
            + ("" if _GROUP_SEP in args.run_prefix else _GROUP_SEP + DEFAULT_SEED)
            + "/results.json\n"
        )
        results = load_results(
            args.checkpoints_dir, args.run_prefix, args.pool_seeds, args.metric
        )
        if not results:
            print("No completed fold results found.")
            return
        print_table(args.run_prefix, results)
        return

    menu = build_menu(args.checkpoints_dir, args.pool_seeds)
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
            results = load_results(
                args.checkpoints_dir, key, args.pool_seeds, args.metric
            )
            if not results:
                print(f"\n## {label}\n  No completed fold results found — skipping.")
                continue
            ap_mu, ap_sd = print_table(key, results)
        elif args.metric is not None:
            print(f"\n## {label}\n  Doc-only baseline has no {args.metric} — skipping.")
            continue
        else:
            ap_mu, ap_sd = print_doc_baseline_row(key)
        summary_rows.append((label, ap_mu, ap_sd))

    if len(summary_rows) > 1:
        name = args.metric or "AP"
        print(f"\n## Cross-variant summary (mean cross {name})")
        col = "" if args.metric else " AP"  # the default header is unchanged
        print(f"{'Variant':<55} {'Mean' + col:>8} {'Std' + col:>8}")
        print("-" * 73)
        for label, mu, sd in summary_rows:
            mu_str = f"{mu:>8.4f}" if mu == mu else f"{'n/a':>8}"
            sd_str = f"{sd:>8.4f}" if not (sd != sd) else f"{'—':>8}"
            print(f"{label:<55} {mu_str} {sd_str}")


if __name__ == "__main__":
    main()
