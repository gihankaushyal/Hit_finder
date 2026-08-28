# ResNet18 Eiger4M Smoke Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `resnet18_resonet.yaml` so it works with `train_asymmetric.py`, create SLURM scripts for smoke test and full LODO run, and submit the smoke test.

**Architecture:** `train_asymmetric.py` requires a `lodo` section in the config; `resnet18_resonet.yaml` currently has `data.cxi_file` instead and will crash immediately. Fix is to replace the config content with a complete `lodo`-compatible config (10 epochs, Eiger4M fold only via `--folds 4`). Full LODO run uses `resnet18_asymmetric.yaml` unchanged.

**Tech Stack:** Python 3.11, PyTorch, SLURM (`sbatch`), W&B, `train_asymmetric.py`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `configs/supervised/resnet18_resonet.yaml` | Replace content | Fix missing `lodo` section; set 10 epochs for smoke test |
| `scripts/submit_resonet_smoketest.sh` | Create | SLURM script for Eiger4M-only smoke test (`--folds 4`) |
| `scripts/submit_asymmetric_lodo.sh` | Create | SLURM script for full 4-fold LODO run |

---

### Task 1: Fix `configs/supervised/resnet18_resonet.yaml`

**Files:**
- Modify: `configs/supervised/resnet18_resonet.yaml`

- [ ] **Step 1: Replace the config content**

Write the following to `configs/supervised/resnet18_resonet.yaml` (replaces existing content entirely):

```yaml
# Smoke-test config: Eiger4M fold only (--folds 4), 10 epochs.
# Run with: python scripts/train_asymmetric.py --config configs/supervised/resnet18_resonet.yaml --folds 4
model:
  backbone: resnet18
  pretrained: true
  num_classes: 2

training:
  epochs: 10
  learning_rate: 1.0e-4
  early_stopping_patience: 10

lodo:
  detector_dirs:
    AGIPD:       /data/bioxfel/user/gihan/Resonet/production/agipd_20k
    JUNGFRAU_4M: /data/bioxfel/user/gihan/Resonet/production/jungfrau_20k
    ePix10k:     /data/bioxfel/user/gihan/Resonet/production/epix10k_20k
    Eiger4M:     /data/bioxfel/user/gihan/Resonet/production/eiger4m_20k
  cxi_pattern: "compressed*.cxi"
  label_key: entry_1/labels/hit

hitfinder:
  backend: pf8
  pf8_threshold: 800.0
  pf8_min_snr: 5.0
  pf8_min_pix_count: 2
  pf8_max_pix_count: 200
  pf8_local_bg_radius: 3
  pf8_min_res: 0
  pf8_max_res: 0
  pf8_use_saturated: false
  gpu_script_path: ""
  gpu_device: cuda

asymmetric:
  hit_frac: 0.5
  hard_neg_max_attempts: 50
  n_cutout_holes: 3
  cutout_hole_size: 32

benchmark:
  aggregation: vote
  min_hit_patches: 3
  patch_stride: 224

wandb:
  tags: [supervised, resnet18, resonet-smoketest]
```

- [ ] **Step 2: Dry-run config load to catch YAML/key errors**

```bash
python -c "
from src.utils.config import load_config
cfg = load_config('configs/supervised/resnet18_resonet.yaml')
import json; print(json.dumps(cfg, indent=2))
"
```

Expected: JSON dump with `lodo`, `hitfinder`, `asymmetric`, `benchmark` keys all present. No `KeyError` or `YAMLError`.

- [ ] **Step 3: Commit**

```bash
git add configs/supervised/resnet18_resonet.yaml
git commit -m "fix: update resnet18_resonet.yaml to lodo-compatible config (10 epochs, Eiger4M smoketest)"
```

---

### Task 2: Create `scripts/submit_resonet_smoketest.sh`

**Files:**
- Create: `scripts/submit_resonet_smoketest.sh`

- [ ] **Step 1: Write the SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=sfx-resonet-smoketest
#SBATCH --partition=general
#SBATCH --qos=grp_cxfl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

source .secrets/wandb.env

python scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_resonet.yaml \
    --folds 4
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x scripts/submit_resonet_smoketest.sh
bash -n scripts/submit_resonet_smoketest.sh && echo "syntax OK"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/submit_resonet_smoketest.sh
git commit -m "feat: add SLURM smoketest script for Eiger4M fold (resnet18_resonet)"
```

---

### Task 3: Create `scripts/submit_asymmetric_lodo.sh`

**Files:**
- Create: `scripts/submit_asymmetric_lodo.sh`

- [ ] **Step 1: Write the SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=sfx-lodo-full
#SBATCH --partition=general
#SBATCH --qos=grp_cxfl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

source .secrets/wandb.env

python scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_asymmetric.yaml
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x scripts/submit_asymmetric_lodo.sh
bash -n scripts/submit_asymmetric_lodo.sh && echo "syntax OK"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/submit_asymmetric_lodo.sh
git commit -m "feat: add SLURM script for full 4-fold LODO run (resnet18_asymmetric)"
```

---

### Task 4: Submit and monitor smoke test

**Files:** none (runtime only)

- [ ] **Step 1: Ensure logs directory exists**

```bash
mkdir -p logs
```

- [ ] **Step 2: Submit the smoke test**

```bash
sbatch scripts/submit_resonet_smoketest.sh
```

Expected output: `Submitted batch job <JOBID>`

- [ ] **Step 3: Monitor job**

```bash
squeue -u $USER
```

Wait for job to reach `R` (running) state, then:

```bash
tail -f logs/<JOBID>.err
```

Expected first lines (within ~60 s of start):
```
Device: cuda
Config: configs/supervised/resnet18_resonet.yaml
Sessions: <N>  total frames: <N>
  AGIPD: <N> sessions
  JUNGFRAU_4M: <N> sessions
  ePix10k: <N> sessions
  Eiger4M: <N> sessions
```

- [ ] **Step 4: Verify smoke test success**

After job completes (exit code 0):
- Check `logs/<JOBID>.err` ends with a results table containing `ap`, `f1`, `auc_roc` — all finite numbers
- Check W&B for a run tagged `resonet-smoketest` with 10 epochs logged

If the job fails, check `logs/<JOBID>.err` for the traceback before proceeding.

---

### Task 5 (Phase 2): Submit full LODO run — only after smoke test passes

**Files:** none (runtime only)

- [ ] **Step 1: Submit full LODO run**

```bash
sbatch scripts/submit_asymmetric_lodo.sh
```

Expected: `Submitted batch job <JOBID>`

- [ ] **Step 2: Monitor**

```bash
squeue -u $USER
tail -f logs/<JOBID>.err
```

Expected final stdout (after all 4 folds):
```
fold_1 | test_detector=AGIPD    | ap=...
fold_2 | test_detector=JUNGFRAU_4M | ap=...
fold_3 | test_detector=ePix10k  | ap=...
fold_4 | test_detector=Eiger4M  | ap=...
mean_ap=...  std_ap=...
```
