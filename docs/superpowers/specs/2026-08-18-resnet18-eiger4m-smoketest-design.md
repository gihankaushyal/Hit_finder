# ResNet18 Eiger4M Smoke Test — Design Spec

**Date:** 2026-08-18
**Branch:** phase-04-pipeline-debug-runner (or new feature branch from main)
**Status:** Approved

## Context

`train_asymmetric.py` is the sole training entry point for Track 1 (Supervised Baseline).
It requires a `lodo` section in the config. `resnet18_resonet.yaml` currently uses a
`data.cxi_file` key instead — it will crash immediately with `KeyError: 'lodo'`.

Goal: fix the config, write a SLURM script, and validate the full preprocessing →
ResNet18 pipeline on Eiger4M data before committing to a full 4-fold LODO run.

---

## Deliverables

### 1. `configs/supervised/resnet18_resonet.yaml` (updated)

Replace current content entirely:

- **Remove** `data.cxi_file` (references non-existent `cxi_merged_25k.cxi`)
- **Add** `lodo.detector_dirs` — all 4 detectors pointing at production data:
  ```
  AGIPD:       /data/bioxfel/user/gihan/Resonet/production/agipd_20k
  JUNGFRAU_4M: /data/bioxfel/user/gihan/Resonet/production/jungfrau_20k
  ePix10k:     /data/bioxfel/user/gihan/Resonet/production/epix10k_20k
  Eiger4M:     /data/bioxfel/user/gihan/Resonet/production/eiger4m_20k
  ```
  All 4 dirs required: the fold guard in `main()` raises `ValueError` if
  `fold["test_detector"]` is absent from `known_detectors`.
- **Add** `lodo.cxi_pattern: "compressed*.cxi"` and `lodo.label_key: entry_1/labels/hit`
- **Add** `hitfinder`, `asymmetric`, `benchmark` sections — identical to
  `resnet18_asymmetric.yaml`
- **Set** `training.epochs: 10` (smoke test; full run uses 100)
- **Set** `wandb.tags: [supervised, resnet18, resonet-smoketest]`

### 2. `scripts/submit_resonet_smoketest.sh` (new)

SLURM job script:
- Partition: `general`, `--qos=grp_cxfl`, `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=128G`, `--time=02:00:00`
- `module load mamba/latest && conda activate sfx-hitfinder`
- `source .secrets/wandb.env`
- `python scripts/train_asymmetric.py --config configs/supervised/resnet18_resonet.yaml --folds 4`

`--folds 4` selects the Eiger4M held-out fold only. Verify fold 4 = Eiger4M by checking
`build_lodo_folds()` in `src/evaluation/benchmark.py` before submitting.

---

## Phase 2 — Full LODO Run

After smoke test passes, submit `resnet18_asymmetric.yaml` unchanged (already correct).
Create `scripts/submit_asymmetric_lodo.sh` mirroring the smoketest script but:
- No `--folds` flag (runs all 4)
- `--time=12:00:00`
- `--job-name=sfx-lodo-full`

---

## Success Criteria

| Check | Smoke test | Full LODO |
|-------|-----------|-----------|
| Job exit code | 0 | 0 |
| W&B run tagged | `resonet-smoketest` | `asymmetric-pipeline` |
| Stdout table | `ap`, `auc_roc`, `f1` all finite | `mean_ap ± std_ap` printed |
| Epochs completed | 10 | up to 100 per fold (early-stop) |

---

## Files Modified / Created

| File | Action |
|------|--------|
| `configs/supervised/resnet18_resonet.yaml` | Replace content |
| `scripts/submit_resonet_smoketest.sh` | Create |
| `scripts/submit_asymmetric_lodo.sh` | Create (Phase 2) |
