# Known pitfalls

## Shared with docs-refresh
Result-table and wording rules are kept in one place: read `.claude/skills/docs-refresh/references/known-pitfalls.md` (sections "Results" and wording) before quoting any number. They cover generations kept apart, sample std, evaluated-split optima versus the deployable `inference_threshold`, float16 cache ("bit-identical" is wrong), and what is not implemented.

## Report-specific
- Re-run `python scripts/aggregate_lodo_results.py --select all`; do not copy figures from notes or old decks. Notes have carried wrong numbers before.
- Single seed means single seed: say so. Do not call a difference real unless it exceeds the seed-to-seed spread, which you compute from the finished seeds with the aggregator (no fixed threshold). Seed coverage changes as jobs finish and differs by arm and fold: list `checkpoints/*-seed4*` and the aggregator output before claiming a mean.
- Vote scores tie heavily, so AP/AUC have tie-aware variants (`ap_vote_tieaware`); say which is quoted.
- Hit fraction in training crops is not the dataset hit rate; do not conflate them.
- "Resolution" means image pixel dimensions here, not crystallographic resolution.
- A digest that prints ERROR for PRs or commits is a failed query, not an empty period: fix it (`gh auth status`) before writing "no PRs".
- Note sections the digest lists as "older or undated headings" may hold new work appended later: open them before deciding they are out of period.

## Links and files
- `reports/`, `CLAUDE.md`, `PLANNING.md`, `MEMORY.md`, `session-tail.md`, the `phase-*-kanban.md` file and `docs/progress_notes.md` are gitignored. Tracked files must not link to them.
- Do not put job IDs or `squeue` output in reports.

## Actions
- Never publish, share, push, merge or edit the kanban without the user's go-ahead. Artifacts stay private.
- Progress notes drift: a section written while a PR or run set was open ("open", "launched", "read the worktree") goes stale when it finishes or is redone. Update `docs/progress_notes.md` (SKILL.md step 5, right after the claims are agreed) every report, or the next digest starts from a false state.
