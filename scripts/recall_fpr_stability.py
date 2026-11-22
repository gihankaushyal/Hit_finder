"""Stability of recall at fixed FPR under the number of negatives (bootstrap).

Reads the per-frame score files written by the test evaluation in each run directory:
scores.npz when results.json is next to it, else scores.inference.npz when
results.inference.json is. A score file newer than its results file (a pass killed
between the two writes) is listed as incomplete and not used.

Resampling unit (--unit). "session" (default): frames of one beamtime session are not
independent (CLAUDE.md constraint 7), so each resample draws the split's sessions with
replacement and then the frames within each drawn session with replacement. "frame":
all positives and the negatives are drawn independently with replacement. At each size
in --sizes below the negative count, n negatives are then drawn with replacement from
the resample's negatives; at the full count the resample is used as drawn. Each
resample is scored once for every target. Reported per (target, size): median and 90%
interval of recall, and the number of resamples with no positive or no negative.

Decision, at the full size: 1e-3 if every run's 90% interval is narrower than
--max-width, else 1e-2 if that holds, else "interval" (report every figure with its
interval). A target also fails when no run has a defined interval, or when a run has
fewer than MIN_EXPECTED_FP / target negatives: below that, recall at the target is
recall above the few highest negatives, which a bootstrap cannot resample beyond. Runs
with one class only, runs without a usable score file, and incomplete score files are
listed, not silently dropped.

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
from src.evaluation.metrics import RECALL_FPR_TARGETS, recall_at_fprs  # noqa: E402
from src.training.inference_results import (  # noqa: E402
    INFERENCE_RESULTS_NAME,
    INFERENCE_SCORES_NAME,
    RESULTS_NAME,
    SCORES_NAME,
)

# (score file, the results file it belongs to); the first usable pair wins.
SCORE_FILES = (
    (SCORES_NAME, RESULTS_NAME),
    (INFERENCE_SCORES_NAME, INFERENCE_RESULTS_NAME),
)
UNITS = ("session", "frame")
DEFAULT_SIZES = (500, 1_000, 2_000, 5_000)
DEFAULT_N_BOOT = 1_000
DEFAULT_MAX_WIDTH = 0.05
INTERVAL = (0.05, 0.95)  # 90% interval
MIN_EXPECTED_FP = 10  # target * n_neg below this: too few negatives for the target


def _resample(
    rng: np.random.Generator, y: np.ndarray, groups: list[np.ndarray] | None
) -> np.ndarray:
    """Indices of one bootstrap resample: frames, or sessions then frames within each."""
    if groups is None:
        pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
        return np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
    chosen = rng.integers(len(groups), size=len(groups))
    return np.concatenate([rng.choice(groups[c], len(groups[c])) for c in chosen])


def stability_rows(
    y_true: np.ndarray,
    score: np.ndarray,
    targets: dict[str, float],
    sizes: list[int],
    n_boot: int,
    seed: int,
    sessions: np.ndarray | None = None,
) -> list[dict]:
    """One row per (target, size): median and 90% interval of bootstrapped recall.

    `sessions` (one id per frame) switches from frame to session resampling. Resamples
    with no positive or no negative have no recall; they are counted (`n_undefined`)
    and left out of the quantiles.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    n_full = int((y_true == 0).sum())
    groups = None
    if sessions is not None:
        sessions = np.asarray(sessions)
        groups = [np.nonzero(sessions == g)[0] for g in np.unique(sessions)]
    rng = np.random.default_rng(seed)
    size_list = sorted({*(n for n in sizes if n < n_full), n_full})
    recalls = np.full((len(size_list), n_boot, len(targets)), np.nan)
    target_values = list(targets.values())
    for b in range(n_boot):
        idx = _resample(rng, y_true, groups)
        y_b, s_b = y_true[idx], score[idx]
        pos_s, neg_s = s_b[y_b == 1], s_b[y_b == 0]
        if len(pos_s) == 0 or len(neg_s) == 0:
            continue  # stays NaN: counted as undefined
        for i, n in enumerate(size_list):
            neg_n = neg_s if n == n_full else rng.choice(neg_s, n)
            y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_n))]
            points = recall_at_fprs(y, np.r_[pos_s, neg_n], target_values)
            recalls[i, b] = [r for r, _ in points]
    rows = []
    for j, t in enumerate(targets):
        for i, n in enumerate(size_list):
            r = recalls[i, :, j]
            defined = r[~np.isnan(r)]
            if len(defined):
                q05, q50, q95 = np.quantile(defined, [INTERVAL[0], 0.5, INTERVAL[1]])
            else:
                q05 = q50 = q95 = float("nan")
            rows.append(
                {
                    "target": t,
                    "n_neg": int(n),
                    "median": float(q50),
                    "q05": float(q05),
                    "q95": float(q95),
                    "n_undefined": int(n_boot - len(defined)),
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


def _score_file(run_dir: Path) -> tuple[Path | None, list[str]]:
    """The first score file whose results file is next to it and not older than it.

    Returns (path or None, the score files present but unusable). The evaluation writes
    the score file first, so one newer than its results file is from a pass killed
    between the two writes, and does not belong to the results beside it.
    """
    incomplete = []
    for score_name, results_name in SCORE_FILES:
        score_path, results_path = run_dir / score_name, run_dir / results_name
        if not score_path.exists():
            continue
        if (
            results_path.exists()
            and results_path.stat().st_mtime >= score_path.stat().st_mtime
        ):
            return score_path, incomplete
        incomplete.append(f"{run_dir.name}/{score_name}")
    return None, incomplete


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--glob", default="*", help="run directories to include")
    p.add_argument("--split", default="cross", choices=["cross", "in_domain"])
    p.add_argument("--score", default="vote", choices=["vote", "max"])
    p.add_argument("--unit", default="session", choices=UNITS, help="resampling unit")
    p.add_argument(
        "--sizes", type=_positive_int, nargs="+", default=list(DEFAULT_SIZES)
    )
    p.add_argument("--n-boot", type=_positive_int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-width", type=float, default=DEFAULT_MAX_WIDTH)
    p.add_argument("--out", type=Path, default=Path("recall_fpr_stability.json"))
    a = p.parse_args(argv)

    rows: list[dict] = []
    runs: dict[str, dict] = {}
    widest = {t: float("nan") for t in RECALL_FPR_TARGETS}  # NaN: no run measured
    undefined: list[str] = []  # runs whose split has no positives or no negatives
    underpowered: dict[str, list[str]] = {t: [] for t in RECALL_FPR_TARGETS}
    missing_scores: list[str] = []
    incomplete: list[str] = []
    for run_dir in sorted(a.checkpoints_dir.glob(a.glob)):
        if not run_dir.is_dir():
            continue
        path, unusable = _score_file(run_dir)
        incomplete += unusable
        if path is None:
            if not unusable:
                missing_scores.append(run_dir.name)
            continue
        with np.load(path) as z:
            y, s = z[f"{a.split}_y_true"], z[f"{a.split}_{a.score}"]
            key = f"{a.split}_session"
            if a.unit == "session" and key not in z:
                raise SystemExit(f"{path} has no {key}; use --unit frame")
            sessions = z[key] if a.unit == "session" else None
        run_rows = stability_rows(
            y, s, RECALL_FPR_TARGETS, a.sizes, a.n_boot, a.seed, sessions
        )
        n_full, n_pos = int((y == 0).sum()), int((y == 1).sum())
        runs[run_dir.name] = {
            "file": path.name,
            "n_pos": n_pos,
            "n_neg": n_full,
            "n_sessions": None if sessions is None else int(len(np.unique(sessions))),
        }
        for t, target in RECALL_FPR_TARGETS.items():
            # A one-class run is listed as undefined below, not as underpowered.
            if n_full and n_pos and n_full * target < MIN_EXPECTED_FP:
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
        raise SystemExit(f"no usable score files under {a.checkpoints_dir}/{a.glob}")

    print(f"Resampling unit: {a.unit}")
    for name, info in runs.items():
        print(
            f"  {name}: {info['n_pos']} positives, {info['n_neg']} negatives"
            + (f", {info['n_sessions']} sessions" if info["n_sessions"] else "")
        )
    print(
        f"\n{'run':<45} {'target':>6} {'n_neg':>7} {'median':>7} {'q05':>7}"
        f" {'q95':>7} {'undef':>6}"
    )
    for r in rows:
        print(
            f"{r['run']:<45} {r['target']:>6} {r['n_neg']:>7} {r['median']:>7.4f}"
            f" {r['q05']:>7.4f} {r['q95']:>7.4f} {r['n_undefined']:>6}"
        )
    if missing_scores:
        print(f"\nNo score file, left out: {', '.join(missing_scores)}")
    if incomplete:
        print(
            "\nScore file without a matching (not older) results file, left out:"
            f" {', '.join(incomplete)}"
        )
    if undefined:
        print(f"\nRecall undefined (one class only), left out: {', '.join(undefined)}")
    for t, names in underpowered.items():
        if names:
            print(
                f"\nFewer than {MIN_EXPECTED_FP}/{t} negatives, so {t} fails:"
                f" {', '.join(names)}"
            )
    decision = decide(widest, a.max_width, underpowered)
    print(f"\nWidest 90% interval at full size: {widest}")
    print(f"Decision (max width {a.max_width}): {decision}")
    result = {
        "split": a.split,
        "score": a.score,
        "unit": a.unit,
        "max_width": a.max_width,
        "n_boot": a.n_boot,
        "seed": a.seed,
        "widest_at_full": {t: (w if w == w else None) for t, w in widest.items()},
        "runs": runs,
        "undefined": undefined,
        "underpowered": underpowered,
        "missing_scores": missing_scores,
        "incomplete": incomplete,
        "decision": decision,
        "rows": rows,
    }
    a.out.write_text(json.dumps(result, indent=2))
    print(f"Written → {a.out}")
    return result


if __name__ == "__main__":
    main()
