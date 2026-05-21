# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

**Title:** Detector-Agnostic Hitfinder for Serial Femtosecond X-ray Crystallography (SFX)
**Institution:** Arizona State University, Biodesign Institute — Fromme Lab
**PI:** Petra Fromme

**Objective:** Train and evaluate a machine-learning classifier that labels SFX diffraction detector images as hit or non-hit, generalizing across AGIPD, JUNGFRAU 4M, ePix10k, and Eiger4M without per-detector retraining.

See `PLANNING.md` for roadmap, open decisions, and risk register.

---

## Fixed Architecture — Do Not Revisit Without Explicit Request

### Modeling: Two Parallel Tracks

**Track 1 — Supervised Baseline**
- ResNet18 first, ResNet50 second
- Fine-tune on labeled hit / non-hit diffraction images
- Pretrained weights from Hugging Face Hub via `timm`

**Track 2 — Self-Supervised (MAE-style)**
- Masked image pretraining on pooled unlabeled XFEL frames
- Backbone: ViT-based encoder (MAE-native) — NOT ResNet
- Attach classification head, fine-tune on labeled data
- Architecture mismatch with Track 1 is intentional and documented

The comparison between Track 1 and Track 2 is itself a scientific contribution.

### Shared Preprocessing Pipeline (both tracks, identical)

```
1. Read HDF5/CXI metadata → identify detector type (AGIPD | JUNGFRAU | ePix10k | Eiger4M)
2. Reborn geometry handler → GeometryList → assemble multi-panel image
3. Global Contrast Normalization (GCN): I_gcn = (I - μ) / (σ + ε)
4. Local Contrast Normalization (LCN): I_lcn(x,y) = (I(x,y) - μ_W(x,y)) / (σ_W(x,y) + ε)
5. Resize to 224×224 (after normalization — never before)
```

**Critical constraints:**
- Detector type is ALWAYS read from metadata. Never infer it from image content.
- Normalization order is GCN → LCN. Never reversed.
- Resize happens last. It is for model compatibility, not detector correction.
- Pipeline must be bit-for-bit identical across both tracks for fair comparison.

---

## Directory Structure

```
sfx-hitfinder/
├── CLAUDE.md
├── PLANNING.md                  # roadmap, open decisions, risks
├── SETUP.md                     # manual install steps (Reborn, SLURM modules)
├── environment.yml              # conda environment definition
├── configs/                     # YAML config files for experiments
│   ├── base.yaml
│   ├── supervised/
│   └── ssl/
├── src/
│   ├── preprocessing/           # Reborn wrappers, GCN, LCN, resize
│   │   ├── io.py                # unified reader: .img (fabio) / .h5 / .cxi (h5py)
│   │   ├── geometry.py          # Reborn geometry handling
│   │   ├── normalize.py         # GCN and LCN implementations
│   │   └── pipeline.py          # full preprocessing pipeline
│   ├── data/
│   │   ├── dataset.py           # UnlabeledDataset (.img/SSL) + SFXDataset (labeled HDF5)
│   │   ├── dataloader.py        # DataLoader factories
│   │   └── synthetic.py         # synthetic data generation
│   ├── models/
│   │   ├── supervised.py        # ResNet18/50 fine-tuning
│   │   └── ssl.py               # MAE encoder + classification head
│   ├── training/
│   │   ├── train_supervised.py
│   │   ├── train_ssl_pretrain.py
│   │   └── train_ssl_finetune.py
│   └── evaluation/
│       ├── metrics.py           # accuracy, precision, recall, F1, AUC
│       └── benchmark.py         # cross-detector evaluation protocol
├── scripts/                     # SLURM job submission scripts
│   ├── submit_supervised.sh
│   ├── submit_ssl_pretrain.sh
│   └── env_check.sh
├── tests/
│   ├── test_preprocessing.py
│   ├── test_io.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_evaluation.py
├── notebooks/                   # exploration only, never source of truth
├── docs/
│   ├── architecture.md
│   ├── data_spec.md
│   └── eval_protocol.md
└── data/                        # symlinks only — no actual data stored here
    ├── raw/                     # symlink → actual storage on Sol
    ├── processed/               # symlink → preprocessed tensor cache
    └── synthetic/
```

---

## Compute Environment — ASU Sol HPC

**Partition:** Dedicated — 8× NVIDIA A100 (80 GB each) | **Scheduler:** SLURM

### Environment Setup

```bash
# Create environment (first time)
conda env create -f environment.yml -n sfx-hitfinder

# Activate (always in this order)
module load mamba/latest
conda activate sfx-hitfinder
```

**Verify CUDA on a compute node before anything else:**

```bash
srun --partition=<your-partition> --gpus=1 --pty bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output: `True` followed by the CUDA version. If `False`, fix the environment before writing any training code.

### Minimal SLURM Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=sfx-hitfinder
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

python src/training/train_supervised.py --config configs/supervised/resnet18.yaml
```

---

## Data Conventions

### File Formats

- Raw detector images: HDF5 (`.h5`) or CXI (`.cxi`) — CXI is HDF5 with a defined schema
- Assembled images (unlabeled SSL data): `.img` — ADSC/MAR format, read via `fabio`; **already assembled, skip Reborn geometry step**
- Geometry files: Reborn-compatible, co-located with or referenced from image files
- Labels: JSON sidecar vs. embedded HDF5 dataset — open decision, label loading raises `NotImplementedError` until resolved
- Train/val/test splits: plaintext `.txt` files listing absolute file paths, one per line

### Detector Types and Expected Image Dimensions (pre-assembly)

| Detector | Facility | Raw Dimensions | Notes |
|----------|----------|----------------|-------|
| AGIPD | EuXFEL | 16 × 512 × 128 px | 16 modules |
| JUNGFRAU 4M | LCLS CXI | 8 × 512 × 1024 px | 8 modules |
| ePix10k | LCLS | varies | multiple configurations |
| Eiger4M | Synchrotron/SSX | 2068 × 2162 px | monolithic |

Post-assembly and post-resize: all images are 224 × 224 × 1 (single channel).

### HDF5 Access Pattern

**Never load entire datasets into RAM.** Open and close the HDF5 file inside `__getitem__`, not `__init__` — required for multiprocessing DataLoader workers:

```python
def __getitem__(self, idx):
    with h5py.File(self.paths[idx], 'r') as f:
        image = f['entry/data/data'][()]   # adjust key to actual schema
    return image, label
```

---

## Key Commands

```bash
# Environment
conda env create -f environment.yml -n sfx-hitfinder
python -c "import torch, h5py, reborn, timm, fabio; print('imports OK')"

# Tests
pytest tests/ -v                                      # full suite
pytest tests/test_preprocessing.py -v                # single module
pytest tests/test_preprocessing.py::test_gcn_order -v  # single test

# Formatting (run before every commit)
black src/ tests/

# SLURM
sbatch scripts/submit_supervised.sh
squeue -u $USER
```

---

## Experiment Tracking

Use **Weights & Biases** (`wandb`) for all training runs. Every run must log:

- Config: model name, backbone, learning rate, batch size, normalization params
- Per-epoch: train loss, val loss, accuracy, precision, recall, F1, AUC
- Detector provenance: which detectors in train set, which in val/test set
- Run tag: `supervised` or `ssl-pretrain` or `ssl-finetune`

No wandb = no run. If wandb is unavailable on a compute node:

```bash
wandb offline
# sync after: wandb sync wandb/offline-run-*/
```

---

## Coding Conventions

- **Python 3.11**, type hints on all public functions
- **Black** for formatting — run before every commit
- Config files in YAML, loaded via a single `load_config()` utility — no hardcoded hyperparameters in training scripts
- All random seeds set explicitly: `torch.manual_seed`, `numpy.random.seed`, `random.seed`
- No magic numbers in source files — named constants or config values only
- Preprocessing steps are individual functions with unit tests, not one monolithic transform

---

## Critical Constraints — Read Before Generating Any Code

1. **Detector type comes from metadata.** Never from image content, filename parsing, or a learned neural layer.

2. **Preprocessing pipeline is shared and fixed.** Any modification to GCN, LCN, resize, or Reborn geometry handling applies to BOTH tracks. Changes require explicit design review — do not patch one track silently.

3. **Normalization before resize.** Always. GCN → LCN → resize. Non-negotiable.

4. **HDF5 files are opened lazily.** Never in `__init__`. Multiprocessing will deadlock otherwise.

5. **"Resolution" means image pixel dimensions** throughout this project — not crystallographic resolution unless the context explicitly concerns diffraction quality.

6. **Real-time deployment is Phase 7.** Do not optimize for inference latency in Phases 3–6.

7. **Cross-detector splits must be clean.** Data from the same beamtime session must not appear in both training and evaluation sets.

8. **Update README.md on every phase transition.** When a phase changes in `PLANNING.md` (status → COMPLETE or CURRENT), immediately update `README.md` to match: advance the phase badge (`![Phase](...)`) and the Project Status table (mark the completed phase ✅ Complete, bold and mark the new phase 🔄 **IN PROGRESS**).

---

## Out of Scope — Never Introduce

- Catalysis, DFT, polymer science, materials property prediction
- Real-time pipeline integration as a current deliverable
- Learned detector-type identification (metadata provides this)
- Crystallographic resolution estimation (separate problem from hit finding)
