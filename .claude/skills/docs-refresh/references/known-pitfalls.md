# Known pitfalls (from the PR #141-#143 refresh)

## Verification
- Audit agents mislabelled v1 numbers as `-v2`, and called some statements verified without opening source. Re-run the aggregator yourself and re-check each claim.
- Claims that turned out wrong when checked: `min_hit_patches` changes the frame score (it is reserved/unused), `preprocess_assembled` still exists (removed in PR #43), cutout is 32x32 (it is 3 holes of 24x24), Eiger4M raw dims (measure: stored (5632, 384), assembled 1687x1687), `stage_frame_cache.sh` takes a plan (it is `sbatch scripts/stage_frame_cache.sh <fold_id>`), `get_hitfinder()` takes no argument (it takes the config dict).
- YAML keys can be dead: `cutout_hole_size: 32` is never read. Say so rather than documenting it as live.
- Geometry bundled inside Reborn (`jungfrau_8_pad_geometry_list`, `eiger4m_64_pad_geometry_list`) is invisible to the frame-cache staleness hash; local `.geom`/`.json` files are hashed.

## Wording
- Frame cache stores float16 frames. Masks and centroids are exact. Never write "bit-identical".
- Std is the sample std (n-1) as printed by `aggregate_lodo_results.py`; a population std gave 0.021 where the sample std is 0.0244.
- `semi_supervised_lodo` raises `NotImplementedError`; do not describe it as available.

## Results
- Keep generations apart: v1 (uncached), v2 (cached), SSL September 2026 (`mae-vits16-fold*`, folds 1-2 predate the PR #34 crop-diversity fix), SSL `-v2` (partial). Never mix prefixes in one table.
- Say which pipeline produced a table. The default note: results are v1-pipeline unless marked v2.
- `cross.f1` and `threshold` are the optimum on the evaluated split (best case). The deployable operating point is the validation-set `inference_threshold`. AP and AUC are threshold-free.
- State the number of finished folds for every arm; label tables that are partial.

## Links and files
- `CLAUDE.md`, `PLANNING.md`, `MEMORY.md`, `session-tail.md`, `phase-05-kanban.md`, `docs/progress_notes.md` are gitignored. Tracked docs must not link to them or cite their section numbers. Say "local, untracked project log" if you must mention one.
- Tracked scripts may still cite `docs/progress_notes.md` (for example `scripts/aggregate_lodo_results.py`). That is a code change: put it in a follow-up PR and list it in the report.

## Git
- `git add` of a gitignored path fails; joined to later commands by `;` the chain kept going and merged PR #141 without its note. Use `&&`.
- `gh pr merge` is refused while CI is still running on the newest commit; watch checks first.
- Do not stage the user's unrelated working-tree changes.
- Gitignore means a feature branch cannot carry local-only files; edit them in place instead.

## Stale phrases to sweep for
`Track 2 (planned)`, `not slow`, `ViT-Base`, `32x32` (as the cutout size), `run_lodo`, `2167`, `2160`, `preprocess_assembled`.
