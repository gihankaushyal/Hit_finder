# CLAUDE.md

## Session Start — Read This, Not MEMORY.md

Read `session-tail.md` (< 800 bytes, auto-updated by Stop hook) for current state.
Run `/resume` for a full context briefing — it loads only the 1–2 archive files
relevant to the current branch (< 2 KB total), not MEMORY.md or PLANNING.md.

Check `phase-05-kanban.md` (< 2 KB) when starting a new task or asking "what's
next" — it holds items the user has already approved in conversation but that
haven't been executed yet (open PRs awaiting merge, pending decisions, in-flight
work), so they survive a context reset instead of living only in chat history.
Add an item here **only** after explicit user approval, never speculatively.
When Phase 5 closes, archive/delete this file and start `phase-0X-kanban.md`.

**Kanban ↔ GitHub issues:** `dashboard/` (local web dashboard, see `dashboard/README.md`)
syncs this file with GitHub issues labelled `task`. Each item gets a
`<!-- gh:#N -->` marker on its first line after the first import — never delete or
edit those markers, and keep new items as column-0 `- [ ]` lines. The first import is
manual (`cd dashboard && npm run kanban:sync -- --dry-run`, then `-- --yes`); never
run the real import without the user's explicit go-ahead.

**Task observer:** at the start of any task-oriented session — any interaction where you
will use tools and produce deliverables — invoke the `task-observer` skill before beginning
work, so skill-improvement opportunities are captured throughout the session. When loading
any skill, check the observation log for OPEN observations tagged to that skill and apply
them to the current work, even if the skill file has not been updated yet. The log lives
under `~/.claude/projects/-data-bioxfel-user-gihan-Hit-finder/skill-observations/` (the
stable project path, never a worktree). Load it independently from this instruction, not
by chaining through another skill. After a compaction the resumed session re-reads this
file, which is what re-invokes it.

Load these sections **on demand only** (not at session start):
- **Compute Environment / SLURM** → only when writing SLURM scripts
- **Directory Structure** → only when creating new files
- **Data Conventions / HDF5 Access** → only when writing data-loading code
- **Key Commands** → on request

Skills: `/resume` (lazy context), `/checkpoint` (save pivot context),
`/watch-slurm` (monitor jobs), `/graphify` (knowledge graph);
project skills in `.claude/skills/`: `docs-refresh`, `progress-report`, `pipeline-check`, `slurm-job`

> **Gotcha:** `PLANNING.md`, `MEMORY.md`, `session-tail.md` and `phase-05-kanban.md` are gitignored and live only on local disk. `PLANNING.md` and `MEMORY.md` were tracked until PR #28: if one goes missing, restore the last tracked version with `git show 7e9b940^:<file> > <file>` and re-apply recent updates (this happened once, restored 2026-09-01). `phase-05-kanban.md` has no tracked history to restore from. `CLAUDE.md` and `.claude/` (except `settings.local.json`) are tracked since 2026-11 and this repo is public: never put a key, token or private note in them.

## Knowledge Graph

Graph exists at `graphify-out/graph.json` (3569 nodes, 7267 edges, 246 communities — rebuilt 2026-11-11; run `graphify update .` after code changes).

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
| Data (meant to be symlinks) | `data/raw/`, `data/processed/`, `data/splits/`, `data/synthetic/` — empty directories in this checkout as of 2026-10-29; production CXI files are read from `/data/bioxfel/user/gihan/Resonet/production/` |
| Detector geometry | `src/preprocessing/geometry.py` loaders; local files in `src/preprocessing/data/` (agipd, epix10k, eiger_resonet) — Eiger4M/Jungfrau use Reborn's bundled loaders |
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
6. Geometric augmentation (training only): random rot90 → random flip
7. Local Contrast Normalization (LCN) per crop/patch: I_lcn(x,y) = (I(x,y) - μ_W(x,y)) / sqrt(σ²_W(x,y) + ε), ε=1e-2 (variance-form ε floors the denominator at 0.1 GCN units — prevents noise explosion on low-variance background patches). LCN is masked: gap/padding/edge pixels (geometry-derived valid-pixel mask, eroded 2 px to drop physically double-size panel-edge pixels) are excluded from μ_W/σ_W and zeroed in the output — prevents halo/ringing at panel boundaries. After GCN, invalid pixels are also filled with 0 (= global mean in GCN units).
8. Cutout augmentation (training only, AFTER LCN): peak-aware random cutout — 3 holes of 24×24, exact 0 in LCN space so they never enter the local statistics; hole positions rejection-sampled to keep an 8 px margin from every hitfinder centroid so Bragg evidence for label=1 is never occluded; a hole is skipped, not force-placed, after 20 failed draws
```

**Critical constraints:**
- Detector type is ALWAYS read from metadata. Never infer it from image content.
- GCN on the full assembled frame before crop/tile. Always: GCN(full frame) → crop/tile → rot90/flip → LCN → cutout. Never reversed.
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
│   (tests/conftest.py holds the shared fixtures: fake Jungfrau assembly, wandb_disabled, ...)
├── configs/                     # YAML config files for experiments
│   ├── base.yaml
│   ├── supervised/              # incl. resnet18_asymmetric{,_jitter}.yaml, resnet18_epix_cache_smoketest{,_crops4,_crops8}.yaml
│   │                            # ssl/: mae_pretrain.yaml, mae_finetune.yaml, mae_finetune_careful.yaml (PR #156 recipe), mae_finetune_careful_diag.yaml, mae_finetune_jitter.yaml, mae_finetune_careful_jitter.yaml (PR #157 positive jitter, opt-in)
│   └── ssl/
├── src/
│   ├── preprocessing/           # Reborn wrappers, GCN, LCN
│   │   ├── io.py                # unified reader: .img (fabio) / .h5 / .cxi (h5py)
│   │   ├── geometry.py          # Reborn geometry handling
│   │   ├── normalize.py         # GCN and LCN implementations
│   │   ├── augment.py           # random_rot90, random_flip, random_cutout, patch_grid, pad_border
│   │   ├── pipeline.py          # geometry assembly + preprocess_eval_patches
│   │   └── data/                # detector geometry files
│   │       ├── agipd.geom           # (Eiger4M and Jungfrau use Reborn's bundled loaders — their local geom files were removed in PR #43)
│   │       ├── eiger_resonet.geom
│   │       └── epix10k.geom
│   ├── hitfinders/              # hitfinder backends (all implemented as of 2026-08-31)
│   │   ├── base.py              # Hitfinder Protocol + MockHitfinder
│   │   ├── numpy_pf8.py         # NumPy PF8 implementation
│   │   ├── pf8.py               # C-extension PF8 wrapper (_pf8_wrap.c/.so, Makefile)
│   │   ├── pf8_python.py        # ssc Cython PF8 bridge
│   │   ├── gpu.py               # GPU wrapper — delegates to user script; CUDA context not fork-safe → num_workers=0 enforced
│   │   └── gpu_pf8.py           # pyFAI OCL_PeakFinder backend
│   ├── data/
│   │   ├── dataset.py           # UnlabeledDataset, MultiFrameCXIDataset, AsymmetricCXIDataset, SSLPretrainCXIDataset
│   │   ├── dataloader.py        # DataLoader factories (asymmetric_loader, ssl_crop_loader, none_collate_fn)
│   │   ├── frame_cache.py       # FrameCache: two-tier on-disk cache for the read→assemble→hitfinder→GCN prefix (Phase 5)
│   │   └── synthetic.py         # synthetic data generation
│   ├── models/
│   │   ├── supervised.py        # ResNet18/50 fine-tuning (build_supervised_model via timm)
│   │   └── ssl.py               # MAE encoder + classification head
│   ├── training/
│   │   ├── train_supervised.py
│   │   ├── train_asymmetric.py      # asymmetric pipeline training (primary Track 1 entry point)
│   │   ├── lodo.py                  # shared LODO fold loop + build_sessions (Track 1 + Track 2)
│   │   ├── recipe.py                # careful fine-tune recipe (warmup + cosine, layer decay, no-wd on norm/bias, drop_path, head init); opt-in via config
│   │   ├── run_naming.py            # shared run-name prefix validation + resume/override checkpoint gate
│   │   ├── override_marker.py       # override_applied.json: stops a requeued SLURM job (same job id, higher restart count) re-running --override-training and deleting the new checkpoint
│   │   ├── wandb_identity.py        # each --override-training attempt gets a new W&B run id (<run>, <run>-o2, ...) stored in <run_dir>/wandb_id.txt; old run tagged `overridden`
│   │   ├── inference_results.py     # result-file + W&B summary helpers for --inference-only
│   │   ├── train_ssl_pretrain.py
│   │   └── train_ssl_finetune.py
│   ├── utils/
│   │   └── config.py            # load_config(): YAML deep-merge (model values win over base.yaml); DEFAULT_SEED=42, SEED_LIMIT, seed_arg() (argparse validator), apply_seed_override() for the --seed flag
│   └── evaluation/
│       ├── metrics.py           # average_precision, auc_roc, f1_at_optimal_threshold
│       └── benchmark.py         # run_patch_agg: patch-grid inference + vote/max aggregation
├── scripts/                     # SLURM job submission + utility scripts
│   ├── templates/slurm_gpu_job.sh     # canonical SLURM template (grp_cxfel / h100 / scg020)
│   ├── aggregate_lodo_results.py      # per-fold results.json → summary table; groups runs by (prefix, suffix); --select all; --pool-seeds averages seeds per fold
│   ├── run_pipeline_debug.py
│   ├── build_frame_cache.py     # builds the on-disk frame cache (read→assemble→hitfinder→GCN, once per frame)
│   ├── plan_cache_staging.py    # computes per-fold NVMe staging plan for the frame cache
│   ├── stage_frame_cache.sh     # rsyncs a fold's cache entries to local NVMe (parallelized via xargs -P 8)
│   ├── plot_hit_frac.py         # wandb.Api()-based plot of realized vs configured hit_frac
│   ├── stage_ssl_data.sh, cleanup_ssl_stage.sh   # stage/cleanup CXI files on local NVMe for SSL pretraining
│   ├── submit_asymmetric_lodo.sh, submit_all_lodo_folds.sh, submit_lodo_parallel.sh, submit_agipd_lodo.sh
│   ├── submit_asymmetric_lodo_all.sh, submit_asymmetric_lodo_fold.sh
│   ├── submit_agipd_smoketest.sh, submit_epix_smoketest.sh, submit_resonet_smoketest.sh
│   ├── submit_{eiger,epix10k,jungfrau}_cache.sh   # per-detector frame-cache build jobs
│   ├── submit_epix_cache_smoketest{,_crops4,_crops8}.sh   # cache smoketests
│   ├── diagnose_crop_shortcut.py, submit_crop_shortcut_{diag,trial,epochs_diag,recipe_diag}.sh   # crop-shortcut diagnostic (position sweep, epoch snapshots, recipe arms)
│   ├── submit_jitter_measurements.sh, measure_crop_stats.py, measure_padding_effect.py   # jitter crop-statistics and padding measurements (PR #157)
│   ├── verify_eiger4m_geometry_match.py
│   ├── submit_ssl_pretrain.sh, submit_ssl_pretrain_all.sh   # MAE pretraining SLURM scripts (Phase 5; grp_cxfel / h100 / scg020)
│   ├── submit_ssl_finetune.sh, submit_ssl_finetune_all.sh   # MAE fine-tune SLURM scripts (Phase 5)
│   ├── env_check.sh
│   ├── probe_hdf5.py            # walk HDF5 key hierarchy for unknown files
│   ├── smoke_test_detector_shapes.py  # quick smoke test for detector assembly shapes
│   ├── visualize_assembled.py   # visualise assembled detector frames for QC
│   └── visualize_mae_recon.py   # MAE reconstruction quality visualizer (Phase 5)
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
│   ├── test_train_asymmetric_resume.py
│   ├── test_vote_aggregation.py
│   ├── test_frame_cache.py      # frame cache manifest staleness + two-tier resolution + bit-exactness gate
│   ├── test_run_naming.py       # run-name patterns + checkpoint collision gate
│   ├── test_aggregate_lodo_results.py   # aggregator grouping by (prefix, suffix)
│   ├── test_ssl_dataset.py      # SSLPretrainCXIDataset multi-crop behaviour
│   ├── test_ssl_model.py
│   ├── test_train_ssl.py, test_ssl_main_requeue.py
│   ├── test_inference_only.py, test_orchestrator_inference.py
│   ├── test_override_marker.py, test_wandb_identity.py, test_wandb_isolation.py
│   ├── test_training_block.py, test_submit_script_parity.py
│   ├── test_seed_override_wiring.py   # --seed override: bounds, per-seed sampling streams, loader seed required, atomic split save (PR #159)
│   ├── test_collect_since.py, test_measure_crop_stats.py, test_measure_padding_effect.py
│   └── test_deprecated_scripts.py
├── notebooks/                   # exploration only, never source of truth
│   ├── lcn_ablation.ipynb              # Phase 3 LCN window ablation study
│   ├── pipeline_debug.ipynb
│   ├── pad_assembler_visual_check.ipynb
│   ├── benchmark_hitfinders.ipynb      # hitfinder backend comparison
│   ├── gpu_hitfinder_scan.ipynb        # pyFAI GPU hitfinder parameter scan
│   ├── eiger4m_pf8_inspection.ipynb    # Eiger4M PF8 peak inspection
│   └── detector_assembly_crosscheck.ipynb  # our Reborn assembly vs Resonet's CrystFEL assembly, per detector
├── docs/
│   ├── architecture.md
│   ├── data_spec.md             # confirmed HDF5 keys per detector
│   ├── eval_protocol.md
│   ├── progress_notes.md (local only, gitignored), detailed_notes.md
│   ├── assets/                  # hero banner, module graphs, pipeline diagram
│   ├── codebase/, archive/, superpowers/
│   └── figures/
│       ├── lcn_ablation/        # ablation comparison PNGs (all 4 detectors)
│       ├── assembly_crosscheck/ # Reborn vs CrystFEL assembly comparison renders
│       ├── crop_shortcut/       # crop-position sweep outputs (PR #157)
│       └── mae_recon/           # MAE reconstruction quality renders (Phase 5)
├── dashboard/                   # local repo dashboard (TypeScript: Hono API + React/Vite); self-contained, not imported by src/ — see dashboard/README.md
├── checkpoints/                 # created at runtime; one sub-dir per run name
│   └── <run-prefix>-fold<N>-seed<S>/   # e.g. resnet18-asymmetric-v2-fold1-seed42, mae-vits16-v2-fold1-seed42; plus asymmetric_splits/, intra_splits/
│       └── best.pt              # saved when val F1 improves; keys: epoch, model_state_dict, val_f1, val_threshold
└── data/                        # meant for symlinks only — no actual data stored here (currently empty directories)
    ├── raw/                     # symlink → actual storage on Sol
    ├── processed/               # symlink → preprocessed tensor cache
    ├── splits/
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
source activate sfx-hitfinder
```

**Verify CUDA on a compute node before anything else:**

```bash
srun --partition=<your-partition> --gpus=1 --pty bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output: `True` followed by the CUDA version. If `False`, fix the environment before writing any training code.

### SLURM Job Script Template

Use `scripts/templates/slurm_gpu_job.sh` as the canonical template for all new job scripts. It reflects verified working parameters for the grp_cxfel allocation on scg020 (H100). Copy it, replace `<jobname>` and the python command, adjust `--time` for the run length, and submit with `sbatch`.

Key parameters (do not change without testing):
- Partition: `general`, QOS: `grp_cxfel`, GRES: `gpu:h100:1`, node: `scg020`
- Activation: `source activate sfx-hitfinder` (after `module load mamba/latest`)
- Always `source .secrets/wandb.env` before the training command

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
| JUNGFRAU 4M | LCLS CXI | 2164×2068 px (pre-assembled canvas) | 8 modules of 514×1030; gap pixels present — geometry via `jungfrau_8_pad_geometry_list()`; assembles to 2173×2173 |
| ePix10k | LCLS | 5632 × 384 px (64 stacked panels) | assembles to 1667×1667 |
| Eiger4M | Synchrotron/SSX | 5632 × 384 px (64 stacked 176×192 panels) | assembles to 1687×1687 via `eiger4m_64_pad_geometry_list()` |

AGIPD assembles to 1273×1273. Shapes measured 2026-10-22 from the production CXI files and the Reborn loaders in `src/preprocessing/geometry.py`.

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
sbatch scripts/submit_ssl_pretrain.sh 1 mae-vits16-v2
squeue -u $USER

# Training — asymmetric pipeline (THE primary Track 1 training path; train_supervised.py is shared utilities only)
source .secrets/wandb.env
python -m src.training.train_asymmetric --config configs/supervised/resnet18_resonet.yaml

# Training — SSL/MAE (Track 2). --run-name-prefix is REQUIRED and versioned (see src/training/run_naming.py);
# an existing checkpoint under the resolved name needs --resume-training or --override-training.
python -u -m src.training.train_ssl_pretrain --config configs/ssl/mae_pretrain.yaml --fold 1 --run-name-prefix mae-vits16-v2
python -u -m src.training.train_ssl_finetune --config configs/ssl/mae_finetune.yaml --fold 1 --run-name-prefix vits16-mae-v2 --pretrain-checkpoint checkpoints/mae-vits16-v2-fold1-seed42/last.pt

# Re-evaluate a finished run without training (Track 1 and SSL fine-tune/probe): needs best.pt; writes
# results.json if absent, else results.inference.json. The orchestrators offer it as the 'inference' answer.
python -m src.training.train_asymmetric --config configs/supervised/resnet18_asymmetric.yaml --run-name-prefix resnet18-asymmetric-v2 --folds 2 --inference-only
python -u -m src.training.train_ssl_finetune --config configs/ssl/mae_finetune.yaml --fold 1 --run-name-prefix vits16-mae-v2 --inference-only

# Seed repeats (PR #159): --seed S (0 <= S < 2**32, default 42) sets init, shuffle AND the per-sample crop/jitter/augment draws; the run
# name carries it (<prefix>-fold<N>-seed<S>). Seed 42 keeps the legacy sampling stream. --resume-training keeps the W&B run id of an
# existing run; use --override-training for a fresh id. Aggregate repeats with: python scripts/aggregate_lodo_results.py --select all [--pool-seeds]
python -m src.training.train_asymmetric --config configs/supervised/resnet18_asymmetric.yaml --run-name-prefix resnet18-asymmetric-v2 --folds 1 --seed 43

# SSL orchestrators (stage → train → cleanup per fold)
bash scripts/submit_ssl_pretrain_all.sh --run-name-prefix mae-vits16-v2
bash scripts/submit_ssl_finetune_all.sh --run-name-prefix vits16-mae-v2 --pretrain-run-prefix mae-vits16-v2

# Frame cache (Phase 5) — caches read→assemble→hitfinder→GCN so it runs once per frame, not once per epoch
# opt-in via the `cache:` block in a run config (disabled by default in configs/base.yaml)
python -m scripts.build_frame_cache --config configs/ssl/mae_pretrain.yaml --cache-root <path>
sbatch scripts/stage_frame_cache.sh <fold_id>   # plans the fold (plan_cache_staging.py), then rsyncs its entries to local NVMe via xargs -P 8
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
- **Training entry points** live in `src/training/`, invoked as `python -m src.training.<name>`. `scripts/` holds SLURM wrappers and one-off utilities only — files there that import from `src/` need a `sys.path.insert` hack that `src/training/` files avoid.

---

## Critical Constraints — Read Before Generating Any Code

1. **Detector type comes from metadata.** Never from image content, filename parsing, or a learned neural layer.

2. **Preprocessing pipeline is shared and fixed.** Any modification to GCN, LCN, crop, or Reborn geometry handling applies to BOTH tracks. Changes require explicit design review — do not patch one track silently.

3. **GCN on the full assembled frame before crop/tile.** Always: GCN(full frame) → crop/tile → rot90/flip → LCN → cutout (cutout after LCN, as of 2026-08-17). There is no resize step — 224×224 is achieved via crop only. Non-negotiable.

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

   Specs and plans under `docs/superpowers/` are gitignored: keep them local, never `git add -f`.

12. **Branch and PR discipline.** Follow this workflow exactly:

   - **Phase start:** Create a `phase-XX` branch from `main` when a new phase begins (e.g. `phase-03`). All phase work lands here first.
   - **Never switch branches in the main checkout** (`/data/bioxfel/user/gihan/Hit_finder`, on `phase-XX`). SLURM jobs and running pipelines read it live, so a checkout can break them.
   - **Feature branches = git worktrees:** for any significant feature, run `git worktree add ../Hit_finder-wt/phase-XX-feature-name -b phase-XX-feature-name phase-XX` from the main checkout and do all work, tests and commits there. Naming is unchanged (`phase-03-normalize`, `phase-03-pipeline`). Small fixes and doc updates may land directly on `phase-XX`.
   - **Worktree setup:** gitignored files are absent in a worktree. Symlink `data/` and `.secrets/` from the main checkout, run `npm install` in `dashboard/` if touching it, and use the same conda env.
   - **Feature PR:** Once tests and CI pass on the feature branch, open a PR targeting `phase-XX` (not `main`) and merge through it. Do not merge directly.
   - **After merge:** `git worktree remove ../Hit_finder-wt/<branch>`. Update the main checkout (`git pull`) only when no queued SLURM job depends on the files being changed.
   - **Phase PR:** When all planned features for the phase are merged into `phase-XX` and the full test suite passes, open a PR from `phase-XX` → `main` to close out the phase.
   - **Naming conventions:** Use lowercase kebab-case for feature names (e.g. `phase-03-lcn-ablation`, not `phase-3-LCN_Ablation`). Phase numbers are always zero-padded to two digits.

---

## Out of Scope — Never Introduce

- Catalysis, DFT, polymer science, materials property prediction
- Real-time pipeline integration as a current deliverable
- Learned detector-type identification (metadata provides this)
- Crystallographic resolution estimation (separate problem from hit finding)

