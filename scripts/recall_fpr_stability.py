"""Stability of recall at fixed FPR under the number of negatives (bootstrap).

Reads the per-frame score files written by the test evaluation (scores.npz, else
scores.inference.npz) in each run directory. For each run and target FPR: draw n
negatives with replacement from the split's negatives, and all positives with
replacement, n_boot times, at each size in --sizes below the negative count and at the
full count; report the median and the 90% interval of recall.

Decision, at the full size: 1e-3 if every run's 90% interval is narrower than
--max-width, else 1e-2 if that holds, else "interval" (report every figure with its
interval). A target also fails when no run has a defined interval, or when a run has
fewer than MIN_EXPECTED_FP / target negatives: below that, recall at the target is
recall above the few highest negatives, which a bootstrap cannot resample beyond.
Runs matched by --glob without a score file are listed, not silently dropped.

    python scripts/recall_fpr_stability.py --glob 'resnet18-asymmetric-v2-fold*' \\
        --out reports/recall_fpr_stability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import RECALL_FPR_TARGETS, recall_at_fpr  # noqa: E402

SCORE_FILES = ("scores.npz", "scores.inference.npz")  # first found wins
DEFAULT_SIZES = (500, 1_000, 2_000, 5_000)
DEFAULT_N_BOOT = 1_000
DEFAULT_MAX_WIDTH = 0.05
INTERVAL = (0.05, 0.95)  # 90% interval
MIN_EXPECTED_FP = 10  # target * n_neg below this: too few negatives for the target


def stability_rows(
    y_true: np.ndarray,
    score: np.ndarray,
    targets: dict[str, float],
    sizes: list[int],
    n_boot: int,
    seed: int,
) -> list[dict]:
    """One row per (target, size): median and 90% interval of bootstrapped recall."""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    pos, neg = score[y_true == 1], score[y_true == 0]
    rng = np.random.default_rng(seed)
    rows = []
    for t, target in targets.items():
        for n in sorted({*(s for s in sizes if s < len(neg)), len(neg)}):
            y = np.r_[np.ones(len(pos)), np.zeros(n)]
            recalls = np.empty(n_boot)
            for b in range(n_boot):
                sample = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, n)]
                recalls[b] = recall_at_fpr(y, sample, target)[0]
            q05, q50, q95 = np.quantile(recalls, [INTERVAL[0], 0.5, INTERVAL[1]])
            rows.append(
                {
                    "target": t,
                    "n_neg": int(n),
                    "median": float(q50),
                    "q05": float(q05),
                    "q95": float(q95),
                }
            )
    return rows


def decide(
    width_at_full: dict[str, float],
    max_width: float,
    underpowered: dict[str, list[str]] | None = None,
) -> str:
    """The strictest target whose widest full-size interval is below max_width.

    A NaN width (no run measured) fails, as does a target with an underpowered run.
    """
    for t in sorted(width_at_full, key=lambda k: RECALL_FPR_TARGETS[k]):
        if (underpowered or {}).get(t):
            continue
        if width_at_full[t] < max_width:  # False for NaN
            return t
    return "interval"


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def _score_file(run_dir: Path) -> Path | None:
    return next((run_dir / n for n in SCORE_FILES if (run_dir / n).exists()), None)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--glob", default="*", help="run directories to include")
    p.add_argument("--split", default="cross", choices=["cross", "in_domain"])
    p.add_argument("--score", default="vote", choices=["vote", "max"])
    p.add_argument(
        "--sizes", type=_positive_int, nargs="+", default=list(DEFAULT_SIZES)
    )
    p.add_argument("--n-boot", type=_positive_int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-width", type=float, default=DEFAULT_MAX_WIDTH)
    p.add_argument("--out", type=Path, default=Path("recall_fpr_stability.json"))
    a = p.parse_args(argv)

    rows: list[dict] = []
    widest = {t: float("nan") for t in RECALL_FPR_TARGETS}  # NaN: no run measured
    undefined: list[str] = []  # runs whose split has no positives or no negatives
    underpowered: dict[str, list[str]] = {t: [] for t in RECALL_FPR_TARGETS}
    missing_scores: list[str] = []
    for run_dir in sorted(a.checkpoints_dir.glob(a.glob)):
        if not run_dir.is_dir():
            continue
        path = _score_file(run_dir)
        if path is None:
            missing_scores.append(run_dir.name)
            continue
        with np.load(path) as z:
            y, s = z[f"{a.split}_y_true"], z[f"{a.split}_{a.score}"]
        run_rows = stability_rows(y, s, RECALL_FPR_TARGETS, a.sizes, a.n_boot, a.seed)
        n_full = int((y == 0).sum())
        for t, target in RECALL_FPR_TARGETS.items():
            if n_full * target < MIN_EXPECTED_FP:
                underpowered[t].append(f"{run_dir.name} ({n_full} negatives)")
        for r in run_rows:
            r.update(run=run_dir.name, file=path.name)
            if r["n_neg"] == n_full:
                width = r["q95"] - r["q05"]
                if np.isnan(width):
                    undefined.append(f"{run_dir.name} ({r['target']})")
                else:
                    w = widest[r["target"]]
                    widest[r["target"]] = width if w != w else max(w, width)
        rows += run_rows
    if not rows:
        raise SystemExit(f"no score files under {a.checkpoints_dir}/{a.glob}")

    print(f"{'run':<45} {'target':>6} {'n_neg':>7} {'median':>7} {'q05':>7} {'q95':>7}")
    for r in rows:
        print(
            f"{r['run']:<45} {r['target']:>6} {r['n_neg']:>7} {r['median']:>7.4f}"
            f" {r['q05']:>7.4f} {r['q95']:>7.4f}"
        )
    if missing_scores:
        print(f"\nNo score file, left out: {', '.join(missing_scores)}")
    if undefined:
        print(f"\nRecall undefined (one class only), left out: {', '.join(undefined)}")
    for t, runs in underpowered.items():
        if runs:
            print(
                f"\nFewer than {MIN_EXPECTED_FP}/{t} negatives, so {t} fails:"
                f" {', '.join(runs)}"
            )
    decision = decide(widest, a.max_width, underpowered)
    print(f"\nWidest 90% interval at full size: {widest}")
    print(f"Decision (max width {a.max_width}): {decision}")
    result = {
        "split": a.split,
        "score": a.score,
        "max_width": a.max_width,
        "n_boot": a.n_boot,
        "seed": a.seed,
        "widest_at_full": {t: (w if w == w else None) for t, w in widest.items()},
        "undefined": undefined,
        "underpowered": underpowered,
        "missing_scores": missing_scores,
        "decision": decision,
        "rows": rows,
    }
    a.out.write_text(json.dumps(result, indent=2))
    print(f"Written → {a.out}")
    return result


if __name__ == "__main__":
    main()
