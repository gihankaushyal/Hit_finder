# Cross-Detector Benchmark Protocol Design

**Date:** 2026-05-13
**Project:** Detector-Agnostic SFX Hitfinder — PhD Thesis, ASU Fromme Lab
**Approach chosen:** B — Protocol + Implementation Spec (combined human-readable doc + executable benchmark module)

---

## Section 1 — Overview & Definitions

### Protocol Variants

Two variants run in parallel. The comparison between them is itself a scientific contribution.

**Strict LODO:** The held-out test detector contributes zero frames to training or pretraining. Pure generalization test.

**Semi-supervised LODO:** The held-out test detector's *unlabeled* frames are added to the SSL pretraining corpus (Track 2 only). Labels and evaluation frames remain held out. Tests whether exposure to unlabeled target-domain data during pretraining improves transfer.

### The 4 Folds

| Fold | Test detector | Train detectors |
|------|--------------|----------------|
| 1 | AGIPD | JUNGFRAU 4M, ePix10k, Eiger4M |
| 2 | JUNGFRAU 4M | AGIPD, ePix10k, Eiger4M |
| 3 | ePix10k | AGIPD, JUNGFRAU 4M, Eiger4M |
| 4 | Eiger4M | AGIPD, JUNGFRAU 4M, ePix10k |

Facility-level separation is the primary guarantee of clean splits: AGIPD is EuXFEL-only, JUNGFRAU 4M and ePix10k are LCLS-only, Eiger4M is Synchrotron/SSX. No fold can have train/test data from the same facility (except Fold 3 vs Fold 2 share LCLS — but the detector types are distinct and collected in separate beamtimes).

---

## Section 2 — Data Split Construction Rules

Within the training detectors for each fold, frames are split **80 / 10 / 10 (train / val / test)** at the **session level**, not the frame level.

### What is a "session"?

A session is one continuous beamtime run, identified by `(detector_type, facility, run_id)`. All frames from the same session stay together — they never span train/val/test boundaries.

**Rationale:** Frames within a run share the same crystal batch, injection conditions, and beam alignment. Random frame-level splits would leak correlated structure and inflate validation metrics.

### Split construction recipe

1. Group frames by `(detector_type, facility, run_id)` — each group is one session unit
2. Sort sessions by frame count (largest first, for balanced bucket filling)
3. Greedily assign sessions to train/val/test buckets until proportions are met
4. Record which session IDs went to which split as a JSON artifact

### Split artifacts

Stored at `data/splits/<fold_id>_<variant>_split.json`. Format:
```json
{
  "fold": 1,
  "variant": "strict_lodo",
  "test_detector": "AGIPD",
  "splits": {
    "session_id_001": "train",
    "session_id_002": "val",
    "session_id_003": "test"
  }
}
```

Split artifacts are committed to git. Experiments must reconstruct splits from these artifacts, not re-run the split algorithm.

### Semi-supervised LODO extension

For the semi-supervised variant only: the held-out test detector's unlabeled frames are appended to the SSL pretraining corpus after split construction. They are never exposed to the supervised classification head or used in evaluation.

---

## Section 3 — Metrics & Reporting

### Primary metric: Average Precision (AP)

Computed per fold from the precision-recall curve. AP integrates over all classification thresholds — threshold-independent and robust to class imbalance. Hit rates in SFX data can be as low as 5%, making AP more informative than accuracy or AUC-ROC alone.

### Secondary metrics

Reported alongside AP but not used for pass/fail decisions:

| Metric | Purpose |
|--------|---------|
| AUC-ROC | Ranking quality across full threshold range |
| F1 at optimal threshold | Single operating-point summary for operational use |
| Precision at optimal threshold | Purity of predicted hits |
| Recall at optimal threshold | Coverage of true hits |

### Aggregation

- Report AP for each of the 4 folds individually
- Report **mean AP ± std** across folds as the headline number
- Table 1 in the thesis chapter: per-detector AP breakdown

### Baseline comparisons

Reported in the same results table:

| Baseline | Description |
|----------|-------------|
| Peakfinder8 | Threshold-based, current industry standard |
| CrystFEL hit criterion | Bragg peak count threshold |
| Within-detector oracle | Trained and tested on the same detector type (upper bound) |

The within-detector oracle is critical: it sets the ceiling for each detector. Detector-agnostic performance is meaningful only relative to what's achievable with detector-specific training.

### Reporting format

One results table per model track (Track 1: ResNet supervised, Track 2: MAE SSL), plus a head-to-head comparison table for Strict LODO vs Semi-supervised LODO on each fold.

---

## Section 4 — Pass Criteria: What "Detector-Agnostic" Means

A model is declared **detector-agnostic** if it satisfies **both** conditions simultaneously on **every fold**:

1. **Relative degradation ≤ X%** — fold AP is within X% of the within-detector oracle AP for that detector
2. **Absolute AP ≥ Y** — fold AP clears a minimum useful operational threshold

### Why both conditions?

Relative degradation alone could declare success on a detector where the oracle itself performs poorly (e.g., a noisy ePix10k dataset where even the in-distribution model scores 0.6 AP). Absolute AP ≥ Y prevents a model from "passing" on a low-quality ceiling.

### Threshold setting

**X and Y are set after Phase 4 baseline results are in.** Fitting thresholds to real data rather than guessing avoids setting a bar that is trivially easy or physically impossible.

Once determined, X and Y are committed to:
- `docs/eval_protocol.md` (narrative)
- `configs/base.yaml` as `detector_agnostic_max_relative_degradation` and `detector_agnostic_min_absolute_ap`

### Protocol failure mode guards

- **Trivial class assignment:** A model that passes by learning only the hit rate prior is caught by reporting full precision-recall curves, not just scalar AP
- **Partial fold failure:** A model that passes 3/4 folds but fails one does NOT pass — the "every fold" requirement is strict. Partial results are reported but not counted as passing.

---

## Section 5 — Implementation Artifacts

### `docs/eval_protocol.md`

Human-readable protocol document. Sections: definitions, fold table, split construction rules, metrics, pass criteria. This is the document cited in the thesis chapter.

### `src/evaluation/benchmark.py`

Executable protocol module. Key public functions:

```python
def build_lodo_folds() -> list[dict]:
    """Returns 4 fold definitions as list of {test_detector, train_detectors}."""

def build_session_stratified_split(
    sessions: list[dict],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, str]:
    """Greedy session-level split. Returns {session_id -> split_name} artifact."""

def run_fold(
    model,
    fold_def: dict,
    split_artifact: dict,
    variant: str,  # "strict_lodo" | "semi_supervised_lodo"
) -> dict:
    """Runs inference on test detector. Returns AP + secondary metrics."""

def run_benchmark(
    model,
    variant: str,
) -> dict:
    """Runs all 4 folds, aggregates, returns results dict."""

def compare_baselines(results: dict) -> str:
    """Formats results table against peakfinder8 and CrystFEL baselines."""
```

### `src/evaluation/metrics.py`

Already stubbed. Implements:
- `average_precision(y_true, y_score) -> float`
- `auc_roc(y_true, y_score) -> float`
- `f1_at_optimal_threshold(y_true, y_score) -> tuple[float, float]`

### `configs/base.yaml` additions

```yaml
benchmark:
  detector_agnostic_max_relative_degradation: null  # set after Phase 4
  detector_agnostic_min_absolute_ap: null           # set after Phase 4
  split_seed: 42
  split_ratios: [0.80, 0.10, 0.10]
```

### Split artifact storage

`data/splits/<fold_id>_<variant>_split.json` — committed to git for reproducibility.

---

## Out of Scope

- Real-time inference benchmarking (Phase 7)
- Multi-crystal or indexing quality metrics (separate problem)
- Per-class breakdown beyond hit/non-hit binary
