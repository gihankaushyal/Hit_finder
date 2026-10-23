# Detailed Results — LODO Benchmark (Track 1 Supervised, Track 2 SSL; 4-fold)

*Full per-fold breakdown referenced from README.md. Last updated: 2026-10-22. All "±" values are the sample standard deviation (n−1) over folds, as printed by `scripts/aggregate_lodo_results.py`.*

**Contents:** [Protocol](#evaluation-protocol) · [Pipeline comparison](#pipeline-comparison) · [Naive baseline](#phase-4-naive-baseline--frame-level-labels-random-crops) · [Asymmetric v1](#asymmetric-pipeline-baseline--hitfinder-guided-crops--all-pr-22-fixes) · [Asymmetric v2](#asymmetric-pipeline-v2--frame-cache-backed-training) · [Track 2 SSL](#track-2--self-supervised-mae-vit-s16) · [Key findings](#key-findings) · [Thresholds](#inference-thresholds)

---

## Evaluation Protocol

**Leave-One-Detector-Out (LODO)** — 4-fold cross-detector generalization benchmark. Each fold holds one detector type entirely out of training and evaluates on all its frames at test time.

| Fold | Train detectors | Held-out (test) |
|------|----------------|-----------------|
| 1 | JUNGFRAU_4M + ePix10k + Eiger4M | AGIPD |
| 2 | AGIPD + ePix10k + Eiger4M | JUNGFRAU_4M |
| 3 | AGIPD + JUNGFRAU_4M + Eiger4M | ePix10k |
| 4 | AGIPD + JUNGFRAU_4M + ePix10k | Eiger4M |

**Metrics:** Average Precision (AP) is the primary metric — area under the precision-recall curve. AUC-ROC and F1 at optimal threshold are secondary. All metrics are computed at the **frame level** via patch-grid vote aggregation (non-overlapping 224×224 tiles, stride 224). A tile counts as a hit when its softmax hit score is above 0.5, and the frame score is `hit_count / n_tiles` (`run_patch_agg` in `src/evaluation/benchmark.py`). The `min_hit_patches` config key is accepted but reserved — it does not currently affect the score.

**Config:** `configs/supervised/resnet18_asymmetric.yaml` — ResNet18, 100 epochs, early-stopping patience 10, AdamW optimizer. The production LODO runs (v2 and later) read the read→assemble→hitfinder→GCN prefix from the on-disk frame cache (`cache:` block; see [architecture](architecture.md)); the v1 runs below did not.

**Data:** Resonet production dataset under `Resonet/production/<detector>_20k` — 5 CXI files (AGIPD, ePix10k, Eiger4M; 4000 frames each) or 10 files (JUNGFRAU 4M; 2000 frames each), i.e. 20k frames per detector, 80k frames total.

**Run naming:** every run is launched with a required, versioned `--run-name-prefix`; checkpoints live in `checkpoints/<prefix>-fold<N>-seed42/`. Re-running an existing name needs `--resume-training` or `--override-training` (`src/training/run_naming.py`; the override is applied once per SLURM job, see `override_marker.py`). Each override attempt gets a fresh W&B run id (`wandb_identity.py`).

**Summarizing results:** `python scripts/aggregate_lodo_results.py` reads `checkpoints/<run>-fold*/results.json` and prints per-variant tables (interactive menu, `--select`, or `--run-prefix`).

---

## Pipeline Comparison

The naive baseline and asymmetric pipeline differ in every stage that determines training crop quality:

| Aspect | Phase 4 Naive Baseline | Asymmetric Pipeline (current) |
|--------|------------------------|-------------------------------|
| Crop strategy | Random 224×224; label inherited from frame | Path A: centroid-centred crop, label=1 / Path B: hard-negative, 50 px clearance, label=0 |
| GCN application | Per-patch after cropping | Full assembled frame before cropping |
| Augmentation order | rot90 → flip → cutout → GCN → LCN | rot90 → flip → LCN → peak-aware cutout |
| LCN ε | std-form, 1e-6 (noise amplification on background) | variance-form, 1e-2 (noise suppressed) |
| Masked LCN | No — gap pixels pollute local stats | Yes, 2 px erosion from panel edges |
| Peak-aware cutout | No — holes could occlude Bragg peaks | Yes, 3 holes of 24×24, 8 px margin around centroids, applied after LCN |
| Crops per frame | One | `asymmetric.crops_per_frame` (default 1); crops are re-drawn every epoch |
| Frame preprocessing | Recomputed every epoch | Optionally served from the frame cache (v2 onward) |

> **Config note:** `resnet18_asymmetric.yaml` sets `n_cutout_holes: 3` and `cutout_hole_size: 32`, but no Python code reads those keys. `AsymmetricCXIDataset` calls `random_cutout(...)` with its defaults (`n_holes=3`, `CUTOUT_HOLE_SIZE_DEFAULT=24` in `src/preprocessing/augment.py`), so the effective hole size is **24 px**.

The naive baseline's frame-level label assignment produced ambiguous training signal: a random crop from a hit frame may contain only background scatter but receives label=1. The asymmetric pipeline assigns labels from crop content (centroid presence), eliminating this ambiguity.

---

## Phase 4 Naive Baseline — frame-level labels, random crops

| Fold | Held-out | Cross AP | Cross AUC | Cross F1 | In-domain AP |
|------|----------|----------|-----------|----------|--------------|
| 1 | AGIPD | 0.5649 | 0.5904 | 0.6661 | 1.0000 |
| 2 | JUNGFRAU_4M | 0.8683 | 0.8156 | 0.7816 | 0.9999 |
| 3 | ePix10k | 0.8825 | 0.8886 | 0.8092 | 1.0000 |
| 4 | Eiger4M | 0.9310 | 0.9138 | 0.8189 | 1.0000 |
| **Mean** | | **0.812 ± 0.167** | | | |

In-domain AP ≈ 1.000 confirms the model perfectly fits the training-detector distribution — capacity is not the bottleneck. The AGIPD cross AP of 0.565 (near-random) dominated the mean and std.

---

## Asymmetric Pipeline Baseline — hitfinder-guided crops + all PR #22 fixes

Full 4-fold 100-epoch results. Config: `configs/supervised/resnet18_asymmetric.yaml`.

### Cross-detector results

| Fold | Held-out | Cross AP | Cross AUC | Cross F1 | Δ AP vs Naive |
|------|----------|----------|-----------|----------|----------------|
| 1 | AGIPD | 0.8074 | 0.8652 | 0.8108 | **+0.242** |
| 2 | JUNGFRAU_4M | 0.8584 | 0.9639 | 0.7570 | −0.010 |
| 3 | ePix10k | 0.8585 | 0.9106 | 0.8943 | −0.024 |
| 4 | Eiger4M | 0.8330 | 0.8596 | 0.8138 | −0.098 |
| **Mean** | | **0.839 ± 0.024** | | | **+0.027** |

### In-domain results (training-detector test set — sanity check)

| Fold | Held-out (train detectors tested) | In-domain AP | In-domain AUC | In-domain F1 |
|------|-----------------------------------|--------------|---------------|--------------|
| 1 | AGIPD held out | 0.8531 | 0.9141 | 0.8391 |
| 2 | JUNGFRAU_4M held out | 0.8085 | 0.8773 | 0.8694 |
| 3 | ePix10k held out | 0.8404 | 0.8998 | 0.8578 |
| 4 | Eiger4M held out | 0.8855 | 0.9377 | 0.8340 |
| **Mean** | | **0.847 ± 0.032** | | |

---

## Asymmetric Pipeline v2 — frame-cache-backed training

Run `resnet18-asymmetric-v2` (`checkpoints/resnet18-asymmetric-v2-fold*-seed42/results.json`, aggregated 2026-10-22). Same preprocessing as v1; the frame cache matches the uncached path through GCN (masks and centroids exactly; frames up to float16 storage rounding, max |Δ| < 0.1 and relative < 1e-3 in `TestBitExactnessAgainstRealData`), so differences come from cheaper epochs under the same early-stopping budget and run-to-run variation, not from a preprocessing change.

### Cross-detector results

| Fold | Held-out | Cross AP | Cross AUC | Cross F1 | Δ AP vs v1 |
|------|----------|----------|-----------|----------|------------|
| 1 | AGIPD | 0.7897 | 0.8473 | 0.7281 | −0.018 |
| 2 | JUNGFRAU_4M | 0.8727 | 1.0000 | 0.7724 | +0.014 |
| 3 | ePix10k | 0.9393 | 1.0000 | 0.8994 | +0.081 |
| 4 | Eiger4M | 0.9980 | 0.9998 | 0.9921 | +0.165 |
| **Mean** | | **0.900 ± 0.090** | **0.962 ± 0.076** | **0.848 ± 0.120** | **+0.061** |

### In-domain results

| Fold | Held-out (train detectors tested) | In-domain AP | In-domain AUC | In-domain F1 |
|------|-----------------------------------|--------------|---------------|--------------|
| 1 | AGIPD held out | 0.8956 | 1.0000 | 0.8153 |
| 2 | JUNGFRAU_4M held out | 0.9254 | 1.0000 | 0.8629 |
| 3 | ePix10k held out | 0.9040 | 0.9999 | 0.8312 |
| 4 | Eiger4M held out | 0.8992 | 1.0000 | 0.8191 |
| **Mean** | | **0.906 ± 0.013** | | |

Mean cross AP rose 0.839 → 0.900, but fold-to-fold spread rose with it (±0.024 → ±0.090): AGIPD is the only fold that regressed, while ePix10k and Eiger4M improved substantially. Full discussion → [progress_notes.md §21](progress_notes.md).

**`crops_per_frame` > 1 has not helped.** An ePix10k cache smoketest (`configs/supervised/resnet18_epix_cache_smoketest{,_crops4,_crops8}.yaml`) found that 4 or 8 crops per frame lowered epoch-3 validation AP/F1 while training loss fell faster, which looks like overfitting to correlated crop-level noise. The default of 1 stays; whether this holds on a full LODO run is open (kanban issue #59).

---

## Track 2 — Self-Supervised (MAE, ViT-S/16)

Encoder: `mae_vit_small_patch16` (embed dim 384, depth 12, 6 heads; decoder 256-dim, depth 4). Pretraining: 400 epochs, batch 256, lr 1.5e-4, mask ratio 0.6, 4 crops per frame. Fine-tune: 100 epochs, lr 5e-5, early-stopping patience 10, on the same asymmetric patch pipeline as Track 1. Linear probe: encoder frozen. Configs: `configs/ssl/mae_{pretrain,finetune}.yaml`.

These numbers are the **September 2026 generation** (pretrain checkpoints `mae-vits16-fold*`, result directories `vits16-mae-{finetune,probe}-fold*-seed42-v2`), run on uncached preprocessing — so the like-for-like Track 1 comparator is **asymmetric v1** (0.839), not v2. The rerun under the current naming standard (`mae-vits16-v2`, `vits16-mae-{finetune,probe}-v2`) had finished only fold 1 of the fine-tune and probe on 2026-10-22 (fine-tune cross AP 0.7728, probe 0.9425 on AGIPD), so it is excluded here.

**Encoder generations are mixed.** By checkpoint timestamp, the fold 1 and 2 pretrained encoders (`mae-vits16-fold{1,2}-seed42/last.pt`, 2026-09-07) predate the crop-diversity fix (PR #34, merged 2026-09-08), while folds 3 and 4 (2026-09-10) include it. The fine-tune and probe stages themselves were all submitted after the fix and use `AsymmetricCXIDataset`, which PR #34 did not touch. `docs/progress_notes.md` §15.7 describes this table as post-fix; `PLANNING.md` records the split above. The fold-1 fine-tune (AP 0.752, best epoch 1) and fold-2 in-domain AP (0.785) are the weakest figures and are the ones the older encoders could explain (n = 2, not proven).

**Fine-tuning peaks very early.** Full fine-tuning at lr 5e-5 reached its best validation F1 within a few epochs (best epochs: fold 1 → 1, fold 2 → 2, fold 4 → 10, fold 3 → 26) and early stopping (patience 10) then ended the run, whereas the frozen-encoder probes trained productively for 48–77 epochs. The fine-tune arm may therefore be under-trained relative to the probe (risk B4 in `PLANNING.md`).

### Cross-detector results

| Fold | Held-out | Fine-tune AP | Fine-tune AUC | Fine-tune F1 | Probe AP | Probe AUC | Probe F1 |
|------|----------|--------------|---------------|--------------|----------|-----------|----------|
| 1 | AGIPD | 0.7519 | 0.8074 | 0.7280 | 0.8415 | 0.8681 | 0.8090 |
| 2 | JUNGFRAU_4M | 0.8754 | 0.9286 | 0.7915 | — | — | — |
| 3 | ePix10k | 0.8747 | 0.9116 | 0.7894 | 0.9062 | 0.9329 | 0.9114 |
| 4 | Eiger4M | 0.8402 | 0.8314 | 0.7830 | 0.7397 | 0.7272 | 0.6977 |
| **Mean** | | **0.836 ± 0.058** (4/4) | 0.870 ± 0.059 | 0.773 ± 0.030 | **0.829 ± 0.084** (3/4) | 0.843 ± 0.105 | 0.806 ± 0.107 |

The fold-2 probe was cancelled at about epoch 79 of 100 (SLURM job 63035721) and never produced a `results.json`.

### In-domain results

| Fold | Fine-tune AP | Fine-tune AUC | Fine-tune F1 | Probe AP | Probe AUC | Probe F1 |
|------|--------------|---------------|--------------|----------|-----------|----------|
| 1 | 0.8920 | 0.9301 | 0.8293 | 0.9038 | 0.9341 | 0.9067 |
| 2 | 0.7846 | 0.7938 | 0.7514 | — | — | — |
| 3 | 0.8876 | 0.9337 | 0.8204 | 0.8913 | 0.9148 | 0.8768 |
| 4 | 0.8837 | 0.9372 | 0.8104 | 0.9390 | 0.9558 | 0.9199 |

### Comparison

| Variant | Mean cross AP | Std | Folds |
|---------|---------------|-----|-------|
| ResNet18 asymmetric v1 (Track 1) | 0.8393 | 0.0244 | 4/4 |
| ViT-S/16 MAE fine-tune (Track 2) | 0.8356 | 0.0581 | 4/4 |
| ViT-S/16 MAE linear probe (Track 2) | 0.8291 | 0.0840 | 3/4 |
| *ResNet18 asymmetric v2 (Track 1, cached) — not comparable to the rows above* | *0.8999* | *0.0895* | *4/4* |

The fine-tuned ViT-S matches the supervised ResNet18 on mean cross AP but is less consistent across held-out detectors (AGIPD is the weakest fold: 0.752). The frozen-encoder probe is the best single result on ePix10k (0.906) and the worst on Eiger4M (0.740).

---

## Key Findings

**AGIPD generalization gap closed:** Cross AP rose from 0.565 → 0.807 (+0.242). The naive pipeline likely learned panel-edge artifacts as detector-identity cues; the asymmetric pipeline's centroid-guided crops force learning of Bragg spot features instead.

**Variance collapsed ~7×:** Cross AP std dropped from ±0.167 → ±0.024. The pipeline is now consistently effective across all four detector types, not just three.

**In-domain AP dropped from ~1.0 to ~0.85:** Expected — patch-level labels from the hitfinder are harder to overfit than frame-level labels. The model can no longer memorize "this detector's panel layout = hit".

**Folds 2–4 slight cross AP decline:** JUNGFRAU, ePix10k, and Eiger4M cross AP dropped 0.010–0.098 versus the naive baseline. This trade-off is accepted: the naive baseline's near-perfect in-domain performance (AP ≈ 1.000) was a sign of label leakage, not generalization. The asymmetric pipeline sacrifices that spurious ceiling for more consistent cross-detector performance.

---

## Inference Thresholds

Each fold's `results.json` holds two different kinds of threshold, and the tables below keep them apart:

- **`inference_threshold`** (top level) is the F1-maximizing threshold on the **validation set**. It is the only one chosen without looking at the evaluated data, so it is the deployable operating point. It falls back to 0.5 if a checkpoint has none.
- **`cross.threshold` / `in_domain.threshold`** are the F1-maximizing thresholds on **the evaluated split itself** (for `cross`, the held-out detector's own labels). The reported `f1` in the `cross` and `in_domain` blocks is the F1 *at that threshold*, so it is an optimistic upper bound for the operating point, not a score a deployed threshold would reach. AP and AUC do not depend on any threshold.

The threshold is chosen over the unique frame scores. Values are discrete because the frame score is `hit_count / n_tiles`.

**Asymmetric v2 — validation-set `inference_threshold`:**

| Fold | `inference_threshold` |
|------|-----------------------|
| 1 (AGIPD) | 0.0123 |
| 2 (JUNGFRAU_4M) | 0.0204 |
| 3 (ePix10k) | 0.0123 |
| 4 (Eiger4M) | 0.0123 |

**Evaluated-split optimum, asymmetric v1** (recorded in the 2026-08 notes; `cross.threshold` / `in_domain.threshold`):

| Fold | Cross threshold | In-domain threshold |
|------|----------------|---------------------|
| 1 (AGIPD) | 0.080 | 0.012 |
| 2 (JUNGFRAU_4M) | 0.012 | 0.184 |
| 3 (ePix10k) | 0.102 | 0.025 |
| 4 (Eiger4M) | 0.388 | 0.012 |

**Evaluated-split optimum, asymmetric v2:**

| Fold | Cross threshold | In-domain threshold |
|------|----------------|---------------------|
| 1 (AGIPD) | 0.080 | 0.012 |
| 2 (JUNGFRAU_4M) | 0.012 | 0.020 |
| 3 (ePix10k) | 0.020 | 0.012 |
| 4 (Eiger4M) | 0.041 | 0.012 |

The gap between the two kinds is informative: the validation-set threshold is the same low value (0.012–0.020) on every fold, while the held-out-detector optimum ranges from 0.012 to 0.080 (v2) and up to 0.388 for Eiger4M (v1). A single threshold chosen on training-detector validation data therefore does not transfer to every unseen detector, and the cross F1 figures in this document should be read as best-case.
