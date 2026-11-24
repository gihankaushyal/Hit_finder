"""Stability of recall at fixed FPR under the number of negatives (bootstrap).

Reads the per-frame score files written by the test evaluation in each run directory:
scores.npz when results.json is next to it, else scores.inference.npz when
results.inference.json is. A score file newer than its results file (a pass killed
between the two writes) is listed as incomplete and not used.

Two resampling units, both run by default (--unit both):
- "frame": all positives and all negatives drawn independently with replacement.
- "session": frames of one session (one CXI file) may not be independent (CLAUDE.md
  constraint 7), so each resample draws the split's S sessions with replacement, then
  the frames within each drawn session with replacement. Recall's between-session
  spread rests on the sessions that hold positives and the threshold's on those that
  hold negatives, so S below is the smaller of those two counts. With few sessions
  (S = 5 per cross split) the cluster bootstrap underestimates the variance by about
  (S-1)/S, and the between-session spread rests on S-1 degrees of freedom, so the width
  is scaled by sqrt(S/(S-1)) * t(S-1, 0.95) / z(0.95) (1.449 at S = 5, 1.175 at
  S = 10). A run with S below MIN_SESSIONS fails every target.
Each run's decision width at the full size is the wider of the two. At each size in
--sizes below the negative count, n negatives are drawn with replacement from the
original negatives of the drawn units, with the resample's positives; each resample is
read at every target. Reported per (unit, target, size): median and 90% interval of
recall, and the number of resamples with no positive or no negative (left out).

Decision: 1e-3 if every run's decision width is below --max-width, else 1e-2 if that
holds, else "interval" (report every figure with its interval). A target also fails
when no run has a defined width; when a run has fewer than MIN_EXPECTED_FP / target
negatives (below that, recall at the target is recall above the few highest negatives,
which a bootstrap cannot resample beyond); or when the chance that a session resample
draws one class only (computed exactly from each session's class counts, and also
counted among the full-size resamples) is above MAX_UNDEFINED_SHARE (the interval would
describe only the resamples that drew both).
Runs with one class only, runs without a usable score file and incomplete score files
are listed, not silently dropped; the decision states how many runs have a measured
width (a run blocked by a gate still counts there).

--score defaults to the aggregation the evaluation stored in the score file (the score
the result file's recall_at_fpr_* keys use), else "vote".

    python scripts/recall_fpr_stability.py --glob 'resnet18-asymmetric-v2-fold*' \\
        --out reports/recall_fpr_stability.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

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
UNITS = ("both", "frame")
DEFAULT_SIZES = (500, 1_000, 2_000, 5_000)
DEFAULT_N_BOOT = 1_000
DEFAULT_MAX_WIDTH = 0.05
INTERVAL = (0.05, 0.95)  # 90% interval
MIN_EXPECTED_FP = 10  # target * n_neg below this: too few negatives for the target
MAX_UNDEFINED_SHARE = 0.01  # chance of a one-class session resample, above this: fail
MIN_SESSIONS = 5  # fewer sessions than this (unit "both"): no session estimate, fail


def _resample(
    rng: np.random.Generator, y: np.ndarray, groups: list[np.ndarray] | None
) -> tuple[np.ndarray, np.ndarray]:
    """One bootstrap resample: (its frame indices, the original frames it drew from).

    Frame unit (groups None): every positive and negative drawn with replacement; the
    pool is all frames. Session unit: sessions drawn with replacement, then frames
    within each; the pool is the drawn sessions' original frames.
    """
    if groups is None:
        pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        return idx, np.arange(len(y))
    chosen = rng.integers(len(groups), size=len(groups))
    idx = np.concatenate([rng.choice(groups[c], len(groups[c])) for c in chosen])
    return idx, np.concatenate([groups[c] for c in chosen])


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
        idx, pool = _resample(rng, y_true, groups)
        y_b, s_b = y_true[idx], score[idx]
        pos_s, neg_s = s_b[y_b == 1], s_b[y_b == 0]
        if len(pos_s) == 0 or len(neg_s) == 0:
            continue  # stays NaN: counted as undefined
        pool_neg = score[pool][y_true[pool] == 0]
        for i, n in enumerate(size_list):
            neg_n = neg_s if n == n_full else rng.choice(pool_neg, n)
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


def session_scale(n_sessions: int) -> float:
    """Widening of a session-bootstrap width over S sessions; NaN if S < 2.

    sqrt(S/(S-1)) corrects the bootstrap variance bias; t(S-1)/z widens for the
    between-session spread being estimated from S-1 degrees of freedom.
    """
    if n_sessions < 2:
        return float("nan")
    q = INTERVAL[1]
    bias = math.sqrt(n_sessions / (n_sessions - 1))
    return bias * float(student_t.ppf(q, n_sessions - 1) / norm.ppf(q))


def one_class_probability(y_true: np.ndarray, sessions: np.ndarray) -> float:
    """Chance that a session resample (S draws of S sessions) holds one class only.

    A drawn session g with k of its n_g frames positive yields no positive with
    probability (1 - k/n_g)^n_g, as its frames are drawn with replacement regardless of
    class. The S draws are independent, so P(no positive) = (mean_g of that)^S; the same
    for negatives. Both cannot happen at once (every frame has a class), so they add.
    """
    y_true, sessions = np.asarray(y_true), np.asarray(sessions)
    ids = np.unique(sessions)
    no_pos = no_neg = 0.0
    for g in ids:
        y_g = y_true[sessions == g]
        frac = float((y_g == 1).mean())
        no_pos += (1.0 - frac) ** len(y_g)
        no_neg += frac ** len(y_g)
    n = len(ids)
    return (no_pos / n) ** n + (no_neg / n) ** n


def _json_safe(value):
    """NaN floats as None, recursively, so the JSON output is strict."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def decide(
    width_at_full: dict[str, float],
    max_width: float,
    blocked: dict[str, list[str]] | None = None,
) -> str:
    """The strictest target whose widest full-size width is below max_width.

    A NaN width (no run measured) fails, as does a target with a blocked run.
    """
    for t in sorted(width_at_full, key=lambda k: RECALL_FPR_TARGETS[k]):
        if (blocked or {}).get(t):
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
    incomplete: list[str] = []
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


def _full_width(rows: list[dict], target: str, n_full: int) -> tuple[float, int]:
    """(q95 - q05, n_undefined) of the full-size row for `target`."""
    r = next(r for r in rows if r["target"] == target and r["n_neg"] == n_full)
    return r["q95"] - r["q05"], r["n_undefined"]


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--glob", default="*", help="run directories to include")
    p.add_argument("--split", default="cross", choices=["cross", "in_domain"])
    p.add_argument(
        "--score",
        default=None,
        choices=["vote", "max"],
        help="default: the aggregation stored in the score file, else vote",
    )
    p.add_argument(
        "--unit",
        default="both",
        choices=UNITS,
        help="'both': frame and session resampling, the wider decides; 'frame' only",
    )
    p.add_argument(
        "--sizes", type=_positive_int, nargs="+", default=list(DEFAULT_SIZES)
    )
    p.add_argument("--n-boot", type=_positive_int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-width", type=float, default=DEFAULT_MAX_WIDTH)
    p.add_argument("--out", type=Path, default=Path("recall_fpr_stability.json"))
    a = p.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)  # fail before the bootstrap

    targets = RECALL_FPR_TARGETS
    rows: list[dict] = []
    runs: dict[str, dict] = {}
    widest = {t: float("nan") for t in targets}  # NaN: no run measured
    undefined: list[str] = []  # runs whose split has no positives or no negatives
    underpowered: dict[str, list[str]] = {t: [] for t in targets}
    undefined_draws: dict[str, list[str]] = {t: [] for t in targets}
    few_sessions: dict[str, list[str]] = {t: [] for t in targets}
    score_mismatch: list[str] = []
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
            stored = str(z["aggregation"]) if "aggregation" in z else None
            score_name = a.score or stored or "vote"
            if a.score and stored and a.score != stored:
                score_mismatch.append(f"{run_dir.name} (evaluated with {stored})")
            y, s = z[f"{a.split}_y_true"], z[f"{a.split}_{score_name}"]
            key = f"{a.split}_session"
            if a.unit == "both" and key not in z:
                raise SystemExit(f"{path} has no {key}; use --unit frame")
            sessions = z[key] if a.unit == "both" else None
        n_full, n_pos = int((y == 0).sum()), int((y == 1).sum())
        by_unit = {"frame": stability_rows(y, s, targets, a.sizes, a.n_boot, a.seed)}
        n_sessions = n_pos_sessions = n_neg_sessions = None
        if sessions is not None:
            n_sessions = int(len(np.unique(sessions)))
            n_pos_sessions = int(len(np.unique(sessions[y == 1])))
            n_neg_sessions = int(len(np.unique(sessions[y == 0])))
        if sessions is not None and n_full and n_pos:  # one class: listed below
            by_unit["session"] = stability_rows(
                y, s, targets, a.sizes, a.n_boot, a.seed, sessions
            )
        info: dict = {
            "file": path.name,
            "score": score_name,
            "aggregation": stored,
            "n_pos": n_pos,
            "n_neg": n_full,
            "n_sessions": n_sessions,
            "n_pos_sessions": n_pos_sessions,
            "n_neg_sessions": n_neg_sessions,
            "width": {},
        }
        runs[run_dir.name] = info
        for unit, unit_rows in by_unit.items():
            for r in unit_rows:
                r.update(run=run_dir.name, file=path.name, unit=unit, score=score_name)
            rows += unit_rows
        if not (n_full and n_pos):
            undefined.append(run_dir.name)
            continue
        # Sessions that bound the session width: recall rests on the sessions with
        # positives, the threshold at a fixed FPR on those with negatives.
        n_eff, p_one = 0, 0.0
        if sessions is not None and n_pos_sessions and n_neg_sessions:
            n_eff = min(n_pos_sessions, n_neg_sessions)
            p_one = one_class_probability(y, sessions)
            info["p_one_class"] = p_one
            for t in targets:
                if n_eff < MIN_SESSIONS:
                    few_sessions[t].append(
                        f"{run_dir.name} ({n_sessions} sessions,"
                        f" {n_pos_sessions} with positives,"
                        f" {n_neg_sessions} with negatives)"
                    )
                if p_one > MAX_UNDEFINED_SHARE:
                    undefined_draws[t].append(f"{run_dir.name} (P = {p_one:.4f})")
        for t, target in targets.items():
            if n_full * target < MIN_EXPECTED_FP:
                underpowered[t].append(f"{run_dir.name} ({n_full} negatives)")
            widths: dict[str, float | None] = {}
            for unit, unit_rows in by_unit.items():
                # The frame unit draws each class separately, so it is never
                # undefined here. For the session unit the Monte Carlo count backs
                # up the exact one_class_probability above.
                width, n_undef = _full_width(unit_rows, t, n_full)
                if unit == "session":
                    width *= session_scale(n_eff)
                    listed = p_one > MAX_UNDEFINED_SHARE  # already listed above
                    if n_undef > MAX_UNDEFINED_SHARE * a.n_boot and not listed:
                        undefined_draws[t].append(
                            f"{run_dir.name} ({n_undef} of {a.n_boot} resamples)"
                        )
                widths[unit] = None if math.isnan(width) else width
            measured = {u: w for u, w in widths.items() if w is not None}
            used = max(measured, key=lambda u: measured[u]) if measured else None
            info["width"][t] = {**widths, "used": used}
            if used is not None:
                w = widest[t]
                widest[t] = measured[used] if w != w else max(w, measured[used])
    if not rows:
        raise SystemExit(
            f"no usable score files under {a.checkpoints_dir}/{a.glob}"
            f" (no score file: {', '.join(missing_scores) or 'none'};"
            " score file without a matching results file:"
            f" {', '.join(incomplete) or 'none'})"
        )

    unit_text = "frame and session (the wider decides)" if a.unit == "both" else "frame"
    print(f"Resampling: {unit_text}")
    for name, info in runs.items():
        print(
            f"  {name} ({info['file']}, {info['score']} score):"
            f" {info['n_pos']} positives, {info['n_neg']} negatives"
            + (
                f", {info['n_sessions']} sessions ({info['n_pos_sessions']} with"
                " positives)"
                if info["n_sessions"]
                else ""
            )
        )
    print(
        f"\n{'run':<45} {'unit':>7} {'target':>6} {'n_neg':>7} {'median':>7}"
        f" {'q05':>7} {'q95':>7} {'undef':>6}"
    )
    for r in rows:
        print(
            f"{r['run']:<45} {r['unit']:>7} {r['target']:>6} {r['n_neg']:>7}"
            f" {r['median']:>7.4f} {r['q05']:>7.4f} {r['q95']:>7.4f}"
            f" {r['n_undefined']:>6}"
        )
    print("\nDecision width at full size (session scaled by session_scale(S)):")
    for name, info in runs.items():
        for t, w in info["width"].items():
            parts = [
                f"{u} {v:.4f}" for u, v in w.items() if u != "used" and v is not None
            ]
            print(f"  {name} {t}: {', '.join(parts)} -> {w['used']}")
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
    for t, names in undefined_draws.items():
        if names:
            print(
                f"\nChance of a one-class session resample above"
                f" {MAX_UNDEFINED_SHARE:.0%}, so {t} fails: {', '.join(names)}"
            )
    for t, names in few_sessions.items():
        if names:
            print(
                f"\nFewer than {MIN_SESSIONS} sessions with positives, so {t} fails:"
                f" {', '.join(names)}"
            )
    if score_mismatch:
        print(
            f"\n--score {a.score} differs from the evaluated aggregation:"
            f" {', '.join(score_mismatch)}"
        )
    blocked = {
        t: underpowered[t] + undefined_draws[t] + few_sessions[t] for t in targets
    }
    decision = decide(widest, a.max_width, blocked)
    covered = sum(
        any(w["used"] is not None for w in info["width"].values())
        for info in runs.values()
    )
    print(f"\nWidest decision width at full size: {widest}")
    print(
        f"Decision (max width {a.max_width}): {decision}"
        f" ({covered} of {len(runs)} runs with a score file have a measured width)"
    )
    result = {
        "split": a.split,
        "score": a.score,  # None: per run, from the score file (see runs)
        "unit": a.unit,
        "max_width": a.max_width,
        "n_boot": a.n_boot,
        "seed": a.seed,
        "widest_at_full": widest,
        "runs": runs,
        "undefined": undefined,
        "underpowered": underpowered,
        "undefined_draws": undefined_draws,
        "few_sessions": few_sessions,
        "score_mismatch": score_mismatch,
        "missing_scores": missing_scores,
        "incomplete": incomplete,
        "decision": decision,
        "decision_covers": {"runs": covered, "of": len(runs)},
        "rows": rows,
    }
    result = _json_safe(result)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Written → {a.out}")
    return result


if __name__ == "__main__":
    main()
