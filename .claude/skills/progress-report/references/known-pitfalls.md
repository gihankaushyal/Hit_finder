# Known pitfalls

## Numbers
- Re-run `python scripts/aggregate_lodo_results.py --select all`; do not copy figures from notes or old decks. Notes have carried wrong numbers before.
- Keep result generations apart: runs differ by pipeline version and by run-name prefix and suffix (for example `-careful`, `-jitter`; see `src/training/run_naming.py` and the aggregator's grouping). Never mix prefixes in one table; say which pipeline produced a table.
- Std is the sample std (n-1), as the aggregator prints. A fold-level number is not a mean; state the number of finished folds for every arm and label partial tables.
- Single seed means single seed: say so, and do not call a difference real unless it exceeds the seed-to-seed spread, which you compute from the finished seeds with the aggregator (no fixed threshold). Seed coverage changes as jobs finish and differs by arm and fold: list `checkpoints/*-seed4*` and the aggregator output before claiming a mean.
- `cross.f1` / `threshold` are the optimum on the evaluated split; the deployable operating point is the validation-set `inference_threshold`. AP and AUC are threshold-free. Vote scores tie heavily, so AP/AUC have tie-aware variants (`ap_vote_tieaware`); say which is quoted.
- Hit fraction in training crops is not the dataset hit rate; do not conflate them.

## Wording
- The frame cache stores float16; never write "bit-identical" for cached vs live.
- "Resolution" means image pixel dimensions here, not crystallographic resolution.
- Do not describe unimplemented things as available (`semi_supervised_lodo` raises `NotImplementedError`).

## Links and files
- `reports/`, `CLAUDE.md`, `PLANNING.md`, `MEMORY.md`, `session-tail.md`, `phase-05-kanban.md`, `docs/progress_notes.md` are gitignored. Tracked files must not link to them.
- Do not put job IDs or `squeue` output in reports.

## Actions
- Never publish, share, push, merge or edit the kanban without the user's go-ahead. Artifacts stay private.
