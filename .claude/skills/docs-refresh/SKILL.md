---
name: docs-refresh
description: Refresh the Hit_finder reading materials (README.md, docs/detailed_notes.md, architecture.md, data_spec.md, eval_protocol.md, docs/codebase/*, plus the local CLAUDE.md tree and PLANNING.md notes) so they match the current code, configs and results. Runs a full audit with parallel read-only agents, verifies every claim against source, edits, opens a PR into the phase branch, runs /code-review high, and stops before merge. Use when the user asks to update, refresh, sync or audit the docs, codebase references or reading materials, after a batch of merged PRs, or at a phase boundary.
disable-model-invocation: true
dashboard-confirm: true
dashboard-art: docs
---

# docs-refresh

Bring the project's reading materials back in line with the code. The last manual refresh (PRs #141-#143) went well but lost time to avoidable mistakes, so this workflow is built around one idea: **an audit finding is a lead, not a fact.** Every claim gets checked against source before it is written.

Read `references/known-pitfalls.md` before starting; it lists the specific mistakes to avoid. Read `references/doc-map.md` to see which source files each doc describes, and `references/audit-prompt.md` when you spawn agents.

## Scope

Tracked docs: `README.md`, `docs/detailed_notes.md`, `docs/architecture.md`, `docs/data_spec.md`, `docs/eval_protocol.md`, `docs/codebase/{index,modules,data-flow,extension-guide,testing}.md`.
Local only (edit in place, never commit): `CLAUDE.md` (tree, stats), `PLANNING.md` (phase notes).
Stats and results: graph counts, test count, God Nodes table, LODO result tables.

Do a full audit every run. Do not shortcut with `git diff` since the last doc commit; drift also comes from configs, results and measured shapes that diffs miss.

## Workflow

### 1. Preflight

```bash
git status --short && git branch --show-current && squeue -u $USER
```

- Be on the current phase branch (`phase-XX`), up to date with origin. Pending SLURM jobs read the live checkout, so never leave the main checkout on an older branch; the code on your docs branch must equal the phase branch.
- Note unrelated working-tree changes (for example `.gitignore`, `setup_env.sh`, `temp_to_deleted/`) and leave them alone.
- `git switch -c phase-XX-docs-<topic>`.

### 2. Establish facts first

Do this yourself, before any agent, so every agent and every edit uses the same numbers:

- `graphify update .`, then read node/edge/community counts and the God Nodes table from `graphify-out/GRAPH_REPORT.md`.
- `pytest tests/ --collect-only -q | tail -1` for the test count; `ls tests/` for the file list.
- Detector shapes: measure from `src/preprocessing/geometry.py` loaders or a probe of a production CXI, not from older docs (they disagreed before).
- Results: `python scripts/aggregate_lodo_results.py` for each run-prefix generation. Record mean, sample std (n-1), per-fold cross and in-domain AP, folds finished.
- Config values the docs quote (`configs/base.yaml`, `configs/ssl/*.yaml`, `configs/supervised/*.yaml`).

Write these into a short facts note in the scratchpad and hand it to agents.

### 3. Parallel audit (read-only)

Spawn one agent per group, in a single message, using the template in `references/audit-prompt.md`:

1. README.md + docs/detailed_notes.md
2. docs/architecture.md + docs/eval_protocol.md + docs/data_spec.md
3. docs/codebase/* (five files)
4. CLAUDE.md tree and stats vs the filesystem

Agents only report; they do not edit (subagents cannot write to the project directory reliably, and unverified edits are how wrong claims got in last time). Each returns claims as `doc:line | documented | observed | source:line | verified/unverified`.

### 4. Verify, then edit

For every reported claim, re-check it yourself (grep, read, run, measure) before editing. Drop claims you cannot confirm and say so in the final report. Then edit in place, keeping each doc's structure; do not rewrite wholesale.

Conventions to hold across docs:
- Pipeline order is fixed (CLAUDE.md): GCN on full frame, crop/tile, rot90/flip, masked LCN, peak-aware cutout (training only).
- State precise claims: the frame cache stores float16 frames with exact masks and centroids, so "bit-identical" is wrong.
- Update "Last updated"/version lines; bump `eval_protocol.md` version when its content changes.

### 5. Result tables

Follow `references/known-pitfalls.md` section "Results". In short: keep generations separate (v1 uncached, v2 cached, SSL September 2026, SSL `-v2`), state the pipeline version, use sample std, label `cross.f1`/`threshold` as evaluated-split optima versus the deployable validation-set `inference_threshold`, and flag any arm with fewer than four finished folds.

### 6. Check links and symbols

```bash
python .claude/skills/docs-refresh/scripts/check_docs.py
```

It exits 1 if a tracked doc links to a gitignored file (CLAUDE.md, PLANNING.md, MEMORY.md, session-tail.md, docs/progress_notes.md, anything `git check-ignore` reports), or if a relative link or backticked repo path does not exist. A stale phrase from the pitfalls list reappearing only prints `WARN` and does not change the exit code, so read every `WARN` line and check its context. Also grep every new backticked identifier across `src/ scripts/ tests/`. Fix everything it reports.

### 7. Local-only files

Update the CLAUDE.md directory tree and stats line and the PLANNING.md phase notes in place. They are gitignored: never `git add` them, and never chain a `git add` of an ignored path with `;` (a failure there once let a merge run without its commit).

### 8. Commit, PR, review

- Stage explicit doc paths only (`git add README.md docs/...`), one commit per doc group.
- Trailers on every commit and the PR body come from the session's attribution reminder (Co-Authored-By and Claude-Session lines; the PR body ends with the Claude Code line and the session URL).
- `git push -u origin <branch>` then `gh pr create --base phase-XX`.
- Run `/code-review high <PR#>` (project rule 10). Fix valid findings in a follow-up commit; defer code changes (for example comments inside scripts) to a separate PR and list them.
- `gh pr checks <PR#> --watch` until CI passes. A merge before CI finishes is refused.
- Use `&&` between dependent git/gh steps, never `;`.

### 9. Stop and report

Report: PR URL, files changed, claims verified and dropped, review findings (fixed / deferred, with reasons), CI status, deferred follow-ups. **Do not merge until the user says so.**

On the user's go: `gh pr merge <PR#> --merge --delete-branch && git switch phase-XX && git pull --ff-only && git fetch --prune`. Confirm the checkout is back on `phase-XX`.
