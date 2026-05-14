# Cross-Detector Leave-One-Detector-Out Benchmark Protocol

**Version:** 1.0 — Phase 1 finalization
**Spec:** `docs/superpowers/specs/2026-05-13-benchmark-protocol-design.md`
**Implementation:** `src/evaluation/benchmark.py`

---

## Overview

This protocol defines how the SFX hitfinder is assessed for detector-agnostic
generalization. The structure is Leave-One-Detector-Out (LODO): train on three
detector types, test on the fourth, repeated for all four detectors.

Two variants run in parallel:

| Variant | Description |
|---------|-------------|
| **Strict LODO** | Held-out detector contributes zero frames to any training step |
| **Semi-supervised LODO** | Held-out detector unlabeled frames added to MAE pretraining only; labels and test frames remain held out |

The comparison between variants is a scientific contribution: it tests whether
unlabeled target-domain exposure during SSL pretraining improves cross-detector
transfer.

---

## The Four Folds

| Fold | Test Detector | Train Detectors | Facility separation |
|------|--------------|----------------|---------------------|
| 1 | AGIPD | JUNGFRAU 4M, ePix10k, Eiger4M | EuXFEL vs LCLS + Synchrotron |
| 2 | JUNGFRAU 4M | AGIPD, ePix10k, Eiger4M | LCLS-CXI vs EuXFEL + LCLS + Synchrotron |
| 3 | ePix10k | AGIPD, JUNGFRAU 4M, Eiger4M | LCLS vs EuXFEL + LCLS-CXI + Synchrotron |
| 4 | Eiger4M | AGIPD, JUNGFRAU 4M, ePix10k | Synchrotron/SSX vs EuXFEL + LCLS |

Facility-level separation is the primary guarantee of clean splits. JUNGFRAU 4M
and ePix10k share LCLS but run in separate beamtimes with independent samples.

---

## Data Split Construction

Within the three training detectors, sessions are split **80 / 10 / 10**
(train / val / in-domain test). Splits happen at the **session level** — all
frames from one beamtime run stay together.

A session is identified by `(detector_type, facility, run_id)`.

**Algorithm (implemented in `build_session_stratified_split`):**

1. Mark all held-out detector sessions as `cross_detector_eval`
2. Sort remaining sessions by frame count (descending)
3. Greedily assign each session to the bucket most below its target proportion

**Reproducibility:** Split seed is `configs/base.yaml benchmark.split_seed`.
Artifacts are saved to `data/splits/fold<N>_<variant>_split.json` and committed
to git.

---

## Metrics

### Primary: Average Precision (AP)

Area under the precision-recall curve. Threshold-independent and robust to
class imbalance. SFX hit rates can be as low as 5%.

### Secondary (reported, not used for pass/fail)

| Metric | Purpose |
|--------|---------|
| AUC-ROC | Ranking quality, full threshold range |
| F1 at optimal threshold | Single operating-point summary |
| Precision / Recall at optimal threshold | Operational breakdown |

### Aggregation

- Per-fold AP for all 4 folds
- **Mean AP +/- std** as headline number
- Baseline comparison via `format_results_table(results, oracle_ap=...)`

### Baselines

| Baseline | Description |
|----------|-------------|
| Peakfinder8 | Threshold-based, current SFX standard |
| CrystFEL hit criterion | Bragg peak count threshold |
| Within-detector oracle | Trained and tested on same detector (upper bound) |

---

## Pass Criteria

A model is **detector-agnostic** only if on **every** fold:

1. **Relative degradation <= X%** — fold AP within X% of the within-detector oracle
2. **Absolute AP >= Y** — fold AP clears a minimum operational threshold

X and Y are set after Phase 4 results. Stored in `configs/base.yaml` as
`benchmark.detector_agnostic_max_relative_degradation` and
`benchmark.detector_agnostic_min_absolute_ap`.

Partial pass (3/4 folds) is reported but does not count as passing.

---

## Reproducibility Checklist

- [ ] `configs/base.yaml` `benchmark.split_seed` recorded
- [ ] Split artifacts committed to `data/splits/`
- [ ] W&B runs tagged `strict_lodo` or `semi_supervised_lodo`
- [ ] Oracle runs logged alongside LODO runs
- [ ] Precision-recall curves exported for all folds
