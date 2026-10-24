# Doc to source-of-truth map

Use this to brief audit agents and to know where to verify a claim.

| Doc | Describes | Verify against |
|-----|-----------|----------------|
| `README.md` | Overview, results, setup, doc table | `scripts/aggregate_lodo_results.py` output, `configs/`, `scripts/`, `docs/` listing, `src/preprocessing/geometry.py` (detector shapes) |
| `docs/detailed_notes.md` | Per-fold result tables, threshold semantics, caveats | aggregator output, `checkpoints/*/results.json`, checkpoint timestamps, `src/evaluation/benchmark.py` |
| `docs/architecture.md` | Two tracks, pipeline, hitfinders, run lifecycle, dashboard | `src/data/dataset.py`, `src/data/frame_cache.py`, `src/models/`, `src/hitfinders/`, `src/training/` (`run_naming.py`, `override_marker.py`, `wandb_identity.py`, `lodo.py`), `configs/ssl/`, `dashboard/README.md` |
| `docs/data_spec.md` | HDF5 keys, sessions, detector shapes, geometry | `src/preprocessing/io.py`, `src/training/lodo.py` (`build_sessions`), `src/preprocessing/geometry.py`, `src/preprocessing/data/` |
| `docs/eval_protocol.md` | LODO folds, metrics, aggregation, result files | `src/evaluation/benchmark.py`, `src/evaluation/metrics.py`, `src/training/inference_results.py`, `configs/base.yaml` (`benchmark:`), `scripts/aggregate_lodo_results.py` |
| `docs/codebase/index.md` | Overview, graph stats, God Nodes | `graphify-out/GRAPH_REPORT.md` |
| `docs/codebase/modules.md` | Per-module exports, importers, scripts table | `src/**`, `scripts/`, `graphify query` for importers |
| `docs/codebase/data-flow.md` | Frame journey, cache, geometry table | `src/data/dataset.py`, `src/data/frame_cache.py`, `src/preprocessing/{io,geometry,pipeline,normalize,augment}.py`, `src/evaluation/benchmark.py` |
| `docs/codebase/extension-guide.md` | Recipes: detector, hitfinder, model, metric, submit scripts, cache | `src/preprocessing/geometry.py`, `src/hitfinders/__init__.py`, `src/models/`, `scripts/submit_*.sh`, `tests/test_submit_script_parity.py` |
| `docs/codebase/testing.md` | Test map, CI, fixtures | `tests/`, `tests/conftest.py`, `.github/workflows/ci.yml`, `dashboard.yml` |
| `CLAUDE.md` (local) | Tree, commands, shapes, stats | `ls`, `git ls-files`, graphify counts |
| `PLANNING.md` (local) | Phase checklist | `phase-05-kanban.md`, `sacct`, merged PRs (`gh pr list --state merged`) |
