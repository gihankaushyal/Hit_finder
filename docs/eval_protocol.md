# Cross-Detector Leave-One-Detector-Out Benchmark Protocol

**Version:** 2.0 — Phase 5 update (2026-10-22); originally 1.0, Phase 1 finalization
**Spec:** `docs/superpowers/specs/2026-05-13-benchmark-protocol-design.md` (original design)
**Implementation:** `src/evaluation/benchmark.py` (splits, patch aggregation, metrics), `src/training/lodo.py` (fold loop, sessions), `src/training/inference_results.py` (result files)

---

## Overview

This protocol defines how the SFX hitfinder is assessed for detector-agnostic
generalization. The structure is Leave-One-Detector-Out (LODO): train on three
detector types, test on the fourth, repeated for all four detectors. Both model
tracks (Track 1 supervised ResNet18, Track 2 MAE ViT-S/16 fine-tune and linear
probe) are evaluated with the identical protocol below.

Two variants are defined:

| Variant | Description | Status |
|---------|-------------|--------|
| **Strict LODO** | Held-out detector contributes zero frames to any training step, including MAE pretraining (`--fold N` excludes fold N's test detector) | **Implemented — all results to date** |
| **Semi-supervised LODO** | Held-out detector unlabeled frames added to MAE pretraining only; labels and test frames remain held out | **Not implemented** — `build_session_stratified_split(variant="semi_supervised_lodo")` raises `NotImplementedError` |

If implemented, the comparison between variants would test whether unlabeled
target-domain exposure during SSL pretraining improves cross-detector transfer.

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

In the Resonet production data a session is one CXI file, identified as
`<detector>_<file stem>` (e.g. `AGIPD_compressed0`) by `build_sessions()` in
`src/training/lodo.py`; its frame count is read from `entry_1/labels/hit`.

**Algorithm (implemented in `build_session_stratified_split`):**

1. Mark all held-out detector sessions as `cross_detector_eval`
2. Sort remaining sessions by frame count (descending)
3. Greedily assign each session to the bucket most below its target proportion

**Reproducibility:** the assignment is fully deterministic from the sorted
session order. `benchmark.split_seed` (42) and `benchmark.split_ratios`
(`[0.80, 0.10, 0.10]`) live in `configs/base.yaml`, but the `seed` argument of
`build_session_stratified_split` is accepted for API compatibility and has no
effect. Each fold's artifact is written to
`checkpoints/asymmetric_splits/fold_<N>.json` when a run starts (not committed;
`data/splits/` is empty).

---

## Metrics

### Primary: Average Precision (AP)

Area under the precision-recall curve. Threshold-independent and robust to
class imbalance. SFX hit rates can be as low as 5%.

### Secondary (reported, not used for pass/fail)

| Metric | Purpose |
|--------|---------|
| AUC-ROC | Ranking quality, full threshold range |
| F1 at optimal threshold | Single operating-point summary; the F1-maximizing threshold over all unique scores **of the evaluated split itself** is returned with it (`threshold`). This is an optimistic, best-case operating point (on `cross`, it uses the held-out detector's labels); AP and AUC-ROC do not depend on a threshold |

`run_on_loader` and `run_patch_agg` return exactly four values: `ap`, `auc_roc`,
`f1` and `threshold`. Precision and recall at the operating point are **not**
computed by the current code.

All headline metrics are **frame-level**. Both are computed twice per fold: on
the held-out detector (`cross`) and on the test split of the training detectors
(`in_domain`, a sanity check that the model fits its training distribution).

### Aggregation

- Per-fold AP for all 4 folds
- **Mean AP +/- std** as headline number, where "std" is the **sample** standard
  deviation (n−1) over folds, as printed by `scripts/aggregate_lodo_results.py`
- `format_results_table(results, oracle_ap=...)` formats the table and supports an
  optional oracle comparison

### Baselines

| Baseline | Description | Status |
|----------|-------------|--------|
| Naive pipeline (random crops, frame-level labels) | Phase 4 comparison point | Reported — see [detailed_notes.md](detailed_notes.md) |
| Peakfinder8 | Threshold-based, current SFX standard | Used inside the pipeline as the label/centroid source; not scored as a standalone frame classifier |
| CrystFEL hit criterion | Bragg peak count threshold | Not run |
| Within-detector oracle | Trained and tested on same detector (upper bound) | Not run — the LODO scripts write only `cross` and `in_domain` blocks |

---

## Frame-level Evaluation

Frames are scored by tiling, not by a single crop (`run_patch_agg` in
`src/evaluation/benchmark.py`):

1. The full assembled frame is GCN-normalized (invalid pixels filled with 0).
2. It is tiled into 224 × 224 patches with `patch_stride` 224 (`patch_grid`), each patch is LCN-normalized with the valid-pixel mask and classified.
3. Patch scores are `softmax[:, 1]`. With `aggregation="vote"` (**the default and the only mode to use for new work**) the frame score is `hit_count / n_patches`, where a patch counts as a hit when its score is > 0.5. `"max"` (max patch score) is kept for backward compatibility. Any other value raises `ValueError`.
4. The deployable decision threshold is the F1-optimal threshold chosen on the **validation set** and stored as `inference_threshold` (0.5 if absent). The `f1` / `threshold` pair reported under `cross` and `in_domain` is instead the optimum on the evaluated split, as described under Metrics.

`min_hit_patches` (default 3) is accepted and recorded in result files but is
reserved: it does not currently change the frame score. Defaults come from the
`benchmark:` block of the configs (`aggregation: vote`, `min_hit_patches: 3`,
`patch_stride: 224`).

## Result Files and Re-evaluation

Each fold writes `checkpoints/<run-prefix>-fold<N>-seed<S>/results.json`:

| Field | Meaning |
|-------|---------|
| `fold_id`, `test_detector` | Fold identity |
| `inference_threshold` | Validation-set F1-optimal threshold (the deployable operating point) |
| `cross` / `in_domain` | `{ap, auc_roc, f1, threshold}` on the held-out detector / training-detector test split; `f1` and `threshold` are the optimum on that split itself |
| `training` *(newer runs)* | `configured_epochs`, `last_epoch`, `best_epoch`, `early_stopped` |
| `inference` *(inference-only passes)* | `aggregation`, `patch_stride`, `min_hit_patches`, `checkpoint_epoch`, `checkpoint_val_f1`, `training_check`, `evaluated_at` |

`--inference-only` (Track 1 `train_asymmetric` and SSL `train_ssl_finetune`,
fine-tune and probe) re-evaluates a finished run from its `best.pt` without
training. It writes `results.json` if the run has none, otherwise
`results.inference.json`, so the original training-time numbers are never
overwritten, and logs `inference/threshold` and
`inference/{in_domain,cross}/<metric>` to the W&B summary
(`src/training/inference_results.py`).

`python scripts/aggregate_lodo_results.py` collects finished folds into
per-variant tables (interactive menu, `--select`, or `--run-prefix`).

## Run Identity

Every run uses a required, versioned `--run-name-prefix` (`src/training/run_naming.py`);
results from different generations (e.g. `resnet18-asymmetric` vs
`resnet18-asymmetric-v2`, `mae-vits16` vs `mae-vits16-v2`) must never be mixed in one
table. An existing checkpoint under the resolved name requires `--resume-training`
or `--override-training`; an override is applied at most once per SLURM job
(`override_marker.py`), and each override attempt logs to a new W&B run id
(`wandb_identity.py`).

## Training Knobs That Affect Comparability

- `asymmetric.hit_frac` (default 0.5, read in `lodo.py`; not an evaluation parameter) sets the hit/miss balance of training crops. `scripts/plot_hit_frac.py` plots the realized against the configured value from W&B.
- `asymmetric.crops_per_frame` (default 1) and the frame cache change the cost of an epoch, not the evaluation protocol.
- Track 2 pretraining must exclude the fold's test detector (`train_ssl_pretrain --fold N`); a pretrain checkpoint is only valid for its own fold.

---

## Pass Criteria

A model is **detector-agnostic** only if on **every** fold:

1. **Relative degradation <= X%** — fold AP within X% of the within-detector oracle
2. **Absolute AP >= Y** — fold AP clears a minimum operational threshold

X and Y were to be set after Phase 4 results. They are stored in `configs/base.yaml` as
`benchmark.detector_agnostic_max_relative_degradation` and
`benchmark.detector_agnostic_min_absolute_ap` and are **still `null`** (as of
2026-10-22), so the pass/fail test is not yet applied; they also depend on the
within-detector oracle, which has not been run.

Partial pass (3/4 folds) is reported but does not count as passing.

---

## Reproducibility Checklist

- [x] `configs/base.yaml` `benchmark.split_seed` recorded (the split itself is deterministic, see above)
- [x] Split artifacts written per fold to `checkpoints/asymmetric_splits/fold_<N>.json` (not committed)
- [ ] W&B runs tagged `strict_lodo` / `semi_supervised_lodo` — not done: the submit scripts tag runs by track (e.g. `supervised,resnet18,asymmetric-pipeline,lodo-cached`, `ssl-pretrain`, `ssl-finetune`)
- [ ] Oracle runs logged alongside LODO runs
- [ ] Precision-recall curves exported for all folds
- [x] Run name prefix recorded and generation kept separate (`--run-name-prefix`)
- [x] `results.json` (with `training` block) kept per fold; re-evaluations go to `results.inference.json`
