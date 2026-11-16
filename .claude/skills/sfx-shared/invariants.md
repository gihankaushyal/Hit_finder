# Invariants

Each repo rule in one line, where it comes from in `CLAUDE.md`, and how to check it.
`CLAUDE.md` is the authority. If this file and `CLAUDE.md` disagree, `CLAUDE.md` wins and
this file gets fixed. The phase-boundary audit (`CLAUDE.md` constraint 9) compares the two.

"Changed files" below means the output of
`git diff --name-only "$PHASE"...HEAD -- '*.py' '*.sh' '*.yaml'`, where `$PHASE` is the
phase branch (for example `phase-05`).

| ID | Rule | `CLAUDE.md` source | How to check |
|---|---|---|---|
| I1 | Detector type comes from file metadata. Never from image content, a file name or a model | Critical constraint 1 | Read every new branch on detector in the changed files: its input must trace back to HDF5/CXI metadata, not to `image.shape`, pixel statistics or `path.name` |
| I2 | Order is GCN(full frame) → crop/tile → rot90/flip → LCN → cutout | Constraint 3; Shared Preprocessing Pipeline | Read the call order in the changed path; `pytest tests/test_normalize.py::TestNormalizationOrder -q` |
| I3 | No resize. 224×224 comes from cropping only | Constraint 3 | `grep -nE "interpolate\(|resize\(|Resize\(|zoom\(|rescale\(" <changed files>` prints nothing |
| I4 | GCN runs on the full assembled frame; after GCN, invalid pixels are filled with 0 | Shared Preprocessing Pipeline, steps 4 and 7 | In the changed path, `gcn(` is called on the assembled frame before any crop or tile |
| I5 | LCN: `lcn_window=9`; `lcn_eps=1e-2` in variance form; masked (invalid pixels excluded from local statistics and zeroed); mask eroded 2 px (`EDGE_EROSION_PX=2`) | Confirmed preprocessing parameters | The values come from config or the named constants; `pytest tests/test_normalize.py -q` |
| I6 | Cutout comes after LCN: 3 holes of 24×24, exact 0, at least 8 px from every hitfinder centroid; a hole is skipped after 20 failed draws | Shared Preprocessing Pipeline, step 8 | `pytest tests/test_augmentation.py -q` |
| I7 | Preprocessing is identical for both tracks. A change to GCN, LCN, crop or geometry applies to both and needs a design review | Critical constraint 2 | The diff changes shared code in `src/preprocessing/`, not a per-track copy; a design note exists |
| I8 | HDF5 files are opened inside `__getitem__`, never in `__init__`; one frame is read (`[i]`), never the whole dataset (`[()]`) | Critical constraint 4; HDF5 Access Pattern | `grep -nE "h5py\.File|\[\(\)\]" <changed files>` and read each hit |
| I9 | Splits are clean: no beamtime session in both training and evaluation data; under leave-one-detector-out, the pretrain checkpoint of a fold never saw the held-out detector | Critical constraint 7 | Split logic goes through `build_sessions` in `src/training/lodo.py`; no random split of frames inside a session |
| I10 | Seeds are set explicitly: `torch`, `numpy`, `random` | Coding Conventions | New entry points call the existing seed helpers (`_set_seeds` in `src/training/train_supervised.py`; `seed_arg` and `apply_seed_override` in `src/utils/config.py`) |
| I11 | No hyperparameter is hard-coded in a training script; configs load through `load_config()`; model YAMLs do not use `defaults: [base]` | Coding Conventions | Read new numeric literals in `src/training/`; `grep -rn "defaults:" configs/` prints nothing |
| I12 | Every training run logs to W&B: config, per-epoch metrics, detector provenance, run tag. No wandb, no run | Experiment Tracking | A new entry point calls `wandb.init` with config and tags; the test fixture `wandb_disabled` is used in tests only |
| I13 | Evaluation tiles the full GCN'd frame (stride 224) and aggregates per frame with `vote` | Shared Preprocessing Pipeline, step 5 | New evaluation code calls `run_patch_agg` with `aggregation="vote"` |
| I14 | The W&B key lives only in `.secrets/wandb.env`. Never inline on a command line, never committed, never written by Claude | Credentials | The secrets scan of `/sfx-security`; SLURM scripts `source .secrets/wandb.env` and do not print the environment |
| I15 | Never switch branches in the main checkout. Significant features use a worktree and a PR into `phase-XX` | Critical constraint 12 | `git -C "$MAIN" branch --show-current` prints `phase-XX`; the guard hook asks on a switch |
| I16 | `docs/superpowers/`, `PLANNING.md`, `MEMORY.md` and the kanban file stay local. Never `git add -f` them | Constraint 11; Session Start | `git ls-files docs/superpowers PLANNING.md MEMORY.md` prints nothing; the guard hook denies the force-add |
| I17 | No inference-latency optimisation before Phase 7 | Critical constraint 6 | Review only: such a change is out of scope, and a reviewer does not ask for one |
| I18 | Training entry points live in `src/training/` and run as `python -m src.training.<name>`; `scripts/` holds SLURM wrappers and one-off utilities | Coding Conventions | A new trainer is not placed under `scripts/` |
