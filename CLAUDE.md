# CLAUDE.md

> **Session start:** Read `MEMORY.md` for current phase status, recent decisions, and known gotchas before doing anything else.

## Knowledge Graph

Graph exists at `graphify-out/graph.json` (501 nodes, 1009 edges, 46 communities — built 2026-06-24; **stale** — run `graphify update .` before querying).

```bash
graphify query "how does the preprocessing pipeline work"
graphify path "SFXDataset" "build_supervised_model"
graphify explain "preprocess_assembled"
graphify update .   # after modifying code — AST-only, no API cost
```

- Prefer `query`/`path`/`explain` over reading raw source for navigation questions.
- If `graphify-out/wiki/index.md` exists, use it for broad architecture review.
- Read `graphify-out/GRAPH_REPORT.md` only for full architecture overview.
- **Subagent write quirk:** spawned subagents cannot write to the project directory — main session must write chunk JSON files manually when running `/graphify` extraction.

### Graph Routing — pick the matching graph for each question

| Topic | Graph | Command |
|-------|-------|---------|
| Our training pipeline, datasets, models, preprocessing, augmentation, hitfinders | `graphify-out/` (this repo) | `graphify query "..."` |
| Reborn internals — PADAssembler, PADGeometry, PADGeometryList, Beam, FrameGetter, detector geometry APIs | `../reborn/graphify-out/` | `cd ../reborn && graphify query "..."` |
| CXI generation, simulation, CXIWriter, geom_parser, Resonet pipeline | `../Resonet/graphify-out/` | `cd ../Resonet && graphify query "..."` |

**Cross-boundary questions** (e.g. "how does our `assemble_only` connect to reborn's `PADAssembler`"): query the reborn graph first for the API side, then the Hit_finder graph for our usage. Always `cd /data/bioxfel/user/gihan/Hit_finder` back to the project root after querying an external graph.

---

## Key Paths — verify before Read

**Rule:** never `Read` a path you have not confirmed exists. `ls`/`find` a scoped
directory first, then Read the confirmed file. Guessing paths for logs, checkpoints,
and data is the largest single source of "File does not exist" errors in this project.
When searching a subtree, scope `find`/`grep` to a specific directory below — a bare
recursive search over the repo root times out.

| What | Where |
|------|-------|
| Training logs | `logs/*.out`, `logs/*.err` (newest: `ls -1t logs/*.err | head`) |
| Checkpoints | `checkpoints/<run>/best.pt` |
| Data (symlinks) | `data/raw/`, `data/processed/`, `data/splits/`, `data/synthetic/` |
| Detector geometry | `src/preprocessing/data/*.geom` (agipd, epix10k, eiger4m) |
| Configs | `configs/base.yaml`, `configs/supervised/`, `configs/ssl/` |
| Geometry source (Resonet) | `/data/bioxfel/user/gihan/Resonet/geoms/*.geom` |

The `/resume` skill runs this discovery automatically at session start.

---

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
2. Reborn geometry handler → PADAssembler → assemble multi-panel image (native resolution)
3. Hitfinder (on-the-fly) → locate Bragg spots; derive hit/non-hit label and centroids
4. Global Contrast Normalization (GCN) on the full assembled frame: I_gcn = (I - μ) / (σ + ε)
5. Crop to 224×224 — training: hitfinder-guided crop (Path A: centred on random Bragg peak → label=1; Path B: random crop with 50 px clearance from all peaks → label=0); eval: patch-grid tiling of the full GCN'd frame (stride=224, score aggregated per frame with vote or max)
6. Augmentation (training only): random rot90 → random flip → peak-aware random cutout (3 holes of 24×24; hole positions rejection-sampled to keep an 8 px margin from every hitfinder centroid so Bragg evidence for label=1 is never occluded; a hole is skipped, not force-placed, after 20 failed draws; holes zero the valid-pixel mask so masked LCN treats them as gaps)
7. Local Contrast Normalization (LCN) per crop/patch: I_lcn(x,y) = (I(x,y) - μ_W(x,y)) / sqrt(σ²_W(x,y) + ε), ε=1e-2 (variance-form ε floors the denominator at 0.1 GCN units — prevents noise explosion on low-variance background patches). LCN is masked: gap/padding/edge pixels (geometry-derived valid-pixel mask, eroded 2 px to drop physically double-size panel-edge pixels) are excluded from μ_W/σ_W and zeroed in the output — prevents halo/ringing at panel boundaries. After GCN, invalid pixels are also filled with 0 (= global mean in GCN units).
```

**Critical constraints:**
- Detector type is ALWAYS read from metadata. Never infer it from image content.
- GCN on the full assembled frame before crop/tile. Always: GCN(full frame) → crop/tile → augment → LCN. Never reversed.
- GCN → LCN order is fixed. Never swap them.
- There is no resize step. 224×224 is achieved via crop only, never downsampling.
- Pipeline must be bit-for-bit identical across both tracks for fair comparison.

---

## Directory Structure

```
sfx-hitfinder/
├── CLAUDE.md
├── MEMORY.md                    # session-start context: current phase, gotchas, next steps
├── PLANNING.md                  # roadmap, open decisions, risks
├── SETUP.md                     # manual install steps (Reborn, SLURM modules)
├── environment.yml              # conda environment definition
├── requirements-ci.txt          # CPU-only deps for GitHub Actions (no CUDA, no Reborn from PyPI)
├── conftest.py                  # inserts project root into sys.path so tests import as `from src.*`
├── configs/                     # YAML config files for experiments
│   ├── base.yaml
│   ├── supervised/
│   └── ssl/
├── src/
│   ├── preprocessing/           # Reborn wrappers, GCN, LCN
│   │   ├── io.py                # unified reader: .img (fabio) / .h5 / .cxi (h5py)
│   │   ├── geometry.py          # Reborn geometry handling
│   │   ├── normalize.py         # GCN and LCN implementations
│   │   ├── augment.py           # random_rot90, random_flip, random_cutout, patch_grid, pad_border
│   │   ├── pipeline.py          # geometry assembly + preprocess_eval_patches
│   │   └── data/                # detector geometry files
│   │       ├── agipd.geom
│   │       ├── eiger4m.geom
│   │       ├── eiger_resonet.geom
│   │       ├── epix10k.geom
│   │       └── jungfrau4m_jf4m_103mm.json
│   ├── hitfinders/              # hitfinder backends
│   │   ├── base.py              # Hitfinder Protocol + MockHitfinder
│   │   ├── numpy_pf8.py         # NumPy PF8 implementation
│   │   ├── pf8.py               # C-extension PF8 wrapper
│   │   └── gpu.py               # GPU backend stub (NotImplementedError)
│   ├── data/
│   │   ├── dataset.py           # UnlabeledDataset, MultiFrameCXIDataset, AsymmetricCXIDataset
│   │   ├── dataloader.py        # DataLoader factories (asymmetric_loader, none_collate_fn)
│   │   └── synthetic.py         # synthetic data generation
│   ├── models/
│   │   ├── supervised.py        # ResNet18/50 fine-tuning (build_supervised_model via timm)
│   │   └── ssl.py               # MAE encoder + classification head
│   ├── training/
│   │   ├── train_supervised.py
│   │   ├── train_ssl_pretrain.py
│   │   └── train_ssl_finetune.py
│   ├── utils/
│   │   └── config.py            # load_config(): YAML deep-merge (model values win over base.yaml)
│   └── evaluation/
│       ├── metrics.py           # average_precision, auc_roc, f1_at_optimal_threshold
│       └── benchmark.py         # run_patch_agg: patch-grid inference + vote/max aggregation
├── scripts/                     # SLURM job submission + utility scripts
│   ├── submit_ssl_pretrain.sh
│   ├── env_check.sh
│   ├── probe_hdf5.py            # walk HDF5 key hierarchy for unknown files
│   ├── smoke_test_detector_shapes.py  # quick smoke test for detector assembly shapes
│   ├── train_asymmetric.py      # asymmetric pipeline training (primary Track 1 script)
│   └── visualize_assembled.py   # visualise assembled detector frames for QC
├── tests/
│   ├── test_preprocessing.py
│   ├── test_io.py
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_normalize.py
│   ├── test_pipeline.py
│   ├── test_models.py
│   ├── test_train_supervised.py
│   ├── test_evaluation.py
│   ├── test_asymmetric_dataset.py
│   ├── test_augmentation.py
│   ├── test_geometry_assembly.py
│   ├── test_hitfinders.py
│   ├── test_patch_eval.py
│   └── test_vote_aggregation.py
├── notebooks/                   # exploration only, never source of truth
│   └── lcn_ablation.ipynb       # Phase 3 LCN window ablation study
├── docs/
│   ├── architecture.md
│   ├── data_spec.md             # confirmed HDF5 keys per detector
│   ├── eval_protocol.md
│   └── figures/
│       └── lcn_ablation/        # ablation comparison PNGs (all 4 detectors)
├── checkpoints/                 # created at runtime; one sub-dir per run name
│   └── <backbone>-seed<N>/
│       └── best.pt              # saved when val F1 improves; keys: epoch, model_state_dict, val_f1, val_threshold
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
mamba env create -f environment.yml -n sfx-hitfinder

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
- Labels: embedded in CXI files at `entry_1/labels/hit` (Resonet format); derived on-the-fly by hitfinder for asymmetric pipeline
- Train/val/test splits: plaintext `.txt` files listing absolute file paths, one per line

### Detector Types and Expected Image Dimensions (pre-assembly)

| Detector | Facility | Raw Dimensions | Notes |
|----------|----------|----------------|-------|
| AGIPD | EuXFEL | 16 × 512 × 128 px | 16 modules |
| JUNGFRAU 4M | LCLS CXI | 2164×2068 px (pre-assembled canvas) | 8 modules of 514×1030; gap pixels present — use `jungfrau4m_crystfel_pad_geometry_list()` |
| ePix10k | LCLS | varies | multiple configurations |
| Eiger4M | Synchrotron/SSX | 2068 × 2162 px | monolithic |

Post-assembly and post-crop: all images are 224 × 224 × 1 (single channel).

**Confirmed preprocessing parameters:** `lcn_window=9` (Phase 3 ablation: window=31 causes panel-edge ringing artifacts; 3/9/15 equivalent on non-hit frames; 9 is the smallest safe choice); `lcn_eps=1e-2` in variance form `sqrt(σ²_W + ε)` (Phase 4 ablation, 2026-08-17: std-form ε=1e-6 amplified readout noise to unit variance on background-only patches — salt-and-pepper static on JUNGFRAU non-hits; 1e-2 suppresses it while preserving Bragg peak amplitude); `EDGE_EROSION_PX=2` valid-pixel mask erosion (Phase 4, 2026-08-17: masked LCN via normalized convolution excludes gap/padding pixels from local stats — fixes gap-boundary halos; measured JF panel-edge pixels ~33% brighter than interior, so the mask is eroded 2 px).

### HDF5 Access Pattern

**Never load entire datasets into RAM.** Open and close the HDF5 file inside `__getitem__`, not `__init__` — required for multiprocessing DataLoader workers:

```python
def __getitem__(self, idx):
    with h5py.File(self.paths[idx], 'r') as f:
        image = f['entry/data/data'][0]    # [0] reads only one frame; [()] loads all N frames into RAM
    return image, label
```

---

## Key Commands

```bash
# Environment
conda env create -f environment.yml -n sfx-hitfinder
python -c "import torch, h5py, reborn, timm, fabio; print('imports OK')"

# Tests (conftest.py adds src/ to sys.path automatically — no PYTHONPATH needed)
pytest tests/ -v                                                                      # full suite
pytest tests/test_normalize.py -v                                                     # single module
pytest tests/test_normalize.py::TestNormalizationOrder -v                             # single test class

# Formatting (run before every commit)
black src/ tests/

# CI dependencies — CPU-only, no Reborn wheel; for local use when full env unavailable
pip install -r requirements-ci.txt

# SLURM
sbatch scripts/submit_ssl_pretrain.sh
squeue -u $USER

# Training — asymmetric pipeline (THE primary Track 1 training path; train_supervised.py is shared utilities only)
source .secrets/wandb.env
python scripts/train_asymmetric.py --config configs/supervised/resnet18_resonet.yaml
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

### Credentials — WANDB_API_KEY

The API key lives in `.secrets/wandb.env` (gitignored — **never commit it, never paste
it inline on a command line**; inline keys get recorded into the permission allowlist and
the session transcript). Before running any training pipeline, load it:

```bash
source .secrets/wandb.env          # exports WANDB_API_KEY
python src/training/train_supervised.py --config configs/supervised/resnet18.yaml
```

The same applies to SLURM: `source .secrets/wandb.env` inside the sbatch job script
before the training command. If `.secrets/wandb.env` is missing, create it from the
template (`export WANDB_API_KEY="..."`) or run `wandb login`; do not hardcode the key
into scripts or configs.

---

## Coding Conventions

- **Python 3.11**, type hints on all public functions
- **Black** for formatting — run before every commit
- Config files in YAML, loaded via a single `load_config()` utility — no hardcoded hyperparameters in training scripts
- `load_config()` (`src/utils/config.py`) always deep-merges `configs/base.yaml` automatically — model values win. Do NOT add `defaults: [base]` to model YAMLs (that is Hydra syntax and pollutes the config dict under plain PyYAML).
- All random seeds set explicitly: `torch.manual_seed`, `numpy.random.seed`, `random.seed`
- No magic numbers in source files — named constants or config values only
- Preprocessing steps are individual functions with unit tests, not one monolithic transform

---

## Critical Constraints — Read Before Generating Any Code

1. **Detector type comes from metadata.** Never from image content, filename parsing, or a learned neural layer.

2. **Preprocessing pipeline is shared and fixed.** Any modification to GCN, LCN, crop, or Reborn geometry handling applies to BOTH tracks. Changes require explicit design review — do not patch one track silently.

3. **GCN on the full assembled frame before crop/tile.** Always: GCN(full frame) → crop/tile → augment → LCN. There is no resize step — 224×224 is achieved via crop only. Non-negotiable.

4. **HDF5 files are opened lazily.** Never in `__init__`. Multiprocessing will deadlock otherwise.

5. **"Resolution" means image pixel dimensions** throughout this project — not crystallographic resolution unless the context explicitly concerns diffraction quality.

6. **Real-time deployment is Phase 7.** Do not optimize for inference latency in Phases 3–6.

7. **Cross-detector splits must be clean.** Data from the same beamtime session must not appear in both training and evaluation sets.

8. **Update README.md on every phase transition.** When a phase changes in `PLANNING.md` (status → COMPLETE or CURRENT), immediately update `README.md` to match: advance the phase badge (`![Phase](...)`) and the Project Status table (mark the completed phase ✅ Complete, bold and mark the new phase 🔄 **IN PROGRESS**).

9. **CLAUDE.md audit at every phase boundary.** Run `/claude-md-management:claude-md-improver` twice per phase:
   - **At phase start** (after creating the `phase-XX` branch) — catch any stale content from the previous phase before writing new code.
   - **At phase end** (before opening the `phase-XX` → `main` PR) — update directory tree, commands, and data conventions to reflect everything added during the phase.

10. **Automated code review on every PR.** Whenever a pull request is created, triggered, or generated, run:
   - `/code-review high` — for feature PRs
   - `/code-review ultra` — for phase-closing PRs (`phase-XX` → `main`)

   Do not consider a PR ready to merge until `/code-review` has run and any high-confidence issues are resolved.

11. **Feature planning workflow.** Before implementing any significant feature, follow this sequence in order:
   1. Run `/superpowers:brainstorming` and `/feature-dev:feature-dev` to explore the design space and identify implementation options.
   2. Use the `AskUserQuestion` tool to gather feedback — ask enough targeted questions to reach ≥95% task clarity before writing any code.
   3. Run `/superpowers:writing-plans` to produce a concrete, confirmed plan based on the answers.
   4. Run `/superpowers:executing-plans` to implement the plan step by step, updating `PLANNING.md` as each checklist item is completed.

   Do not skip brainstorming for "small" features — the sequence applies to all non-trivial work within a phase.

12. **Branch and PR discipline.** Follow this workflow exactly:

   - **Phase start:** Create a `phase-XX` branch from `main` when a new phase begins (e.g. `phase-03`). All phase work lands here first.
   - **Feature branches:** For any significant feature within a phase, cut a `phase-XX-feature-name` branch from `phase-XX` (e.g. `phase-03-normalize`, `phase-03-pipeline`). Small fixes and doc updates may land directly on `phase-XX`.
   - **Feature PR:** Once all tests pass on a `phase-XX-feature-name` branch, open a PR targeting `phase-XX` (not `main`). Do not merge directly — always go through a PR.
   - **Phase PR:** When all planned features for the phase are merged into `phase-XX` and the full test suite passes, open a PR from `phase-XX` → `main` to close out the phase.
   - **Naming conventions:** Use lowercase kebab-case for feature names (e.g. `phase-03-lcn-ablation`, not `phase-3-LCN_Ablation`). Phase numbers are always zero-padded to two digits.

---

## Out of Scope — Never Introduce

- Catalysis, DFT, polymer science, materials property prediction
- Real-time pipeline integration as a current deliverable
- Learned detector-type identification (metadata provides this)
- Crystallographic resolution estimation (separate problem from hit finding)

