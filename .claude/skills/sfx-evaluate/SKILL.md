---
name: sfx-evaluate
description: Evaluation workflow for the Hit_finder repo, in three modes. "code" — is this branch ready for a PR (tests, formatting, risk-tiered review, verified findings, a verdict the guard hook reads). "results" — can these numbers be reported (run inventory, integrity, aggregation, arm comparison against seed spread). "validity" — should this experiment be trusted (splits, seeds, arms, preprocessing, evaluation protocol, calibration provenance). Use before opening a PR, before reporting or presenting numbers, and when asked whether a result is real.
---

# sfx-evaluate

Say first: "Starting the sfx-evaluate workflow (<code|results|validity>)." If you
started it yourself, say that too. No argument: ask which mode.

## Start

1. Read `.claude/skills/sfx-shared/gates.md`, `invariants.md`, `risk-tiers.md` and
   `report-template.md` from the same folder.
2. ```bash
   MAIN="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
   PHASE="$(git -C "$MAIN" branch --show-current)"
   ```
   Repeat these lines at the start of any command that uses the variables.

## Mode: code

Question: is this branch ready for a PR into `$PHASE`?

1. **Clean tree.** `git status --short` must show no tracked change. The report names a
   commit; uncommitted work has none. If the tree is dirty, stop and say so.
2. **Diff and tier.**
   ```bash
   git diff --name-only "$PHASE"...HEAD
   git rev-parse HEAD; git merge-base "$PHASE" HEAD
   ```
   Pick the tier from `risk-tiers.md` and say it with the reason.
3. **Mechanical checks.**
   ```bash
   RUFF="$(command -v ruff || { [ -x "$HOME/.venvs/sfx-lint/bin/ruff" ] && echo "$HOME/.venvs/sfx-lint/bin/ruff"; })"
   MYPY="$(command -v mypy || { [ -x "$HOME/.venvs/sfx-lint/bin/mypy" ] && echo "$HOME/.venvs/sfx-lint/bin/mypy"; })"
   python -m pytest tests/ -q 2>&1 | tail -5
   black --check src/ tests/
   if [ -n "$RUFF" ]; then "$RUFF" check . || true; else echo "ruff: not installed"; fi
   if [ -n "$MYPY" ]; then "$MYPY" --python-executable "$(command -v python)" || true; else echo "mypy: not installed"; fi
   ```
   `ruff` and `mypy` are report-only: their findings are listed and do not change the
   verdict. Each is looked up on `PATH`, then in `$HOME/.venvs/sfx-lint/bin/` (a separate
   venv, so the training environment is untouched); `mypy` reads the packages of the
   `python` on `PATH` through `--python-executable`. "not installed" means the tool was
   not found, never that it found problems. If `src/preprocessing/` changed, read
   `.claude/skills/pipeline-check/SKILL.md` and run its commands.
4. **Review panel.** Run the agents the tier lists, in parallel. Give each the checkout
   path, the range `<merge-base>..<HEAD>` and the tier. Low tier: review inline against
   `invariants.md`.
5. **Verify.** For each finding, read the cited lines yourself and confirm the claim;
   run the single test that would show it when one exists. A finding you cannot confirm
   goes under "Unverified, dropped" with the reason. Report confirmed findings only.
6. **Verdict.**

| Verdict | When |
|---|---|
| `not-ready` | A test fails, `black --check` fails, or a 🔴 finding is confirmed |
| `ready-after-fixes` | No 🔴, but a confirmed 🟡 that the user wants fixed before the PR |
| `ready` | Tests and black pass, no confirmed 🔴, and no 🟡 waiting for a fix |

   Design notes and pre-existing gaps never change the verdict. Ask the user which 🟡
   findings to fix before settling between `ready` and `ready-after-fixes`.
7. **Report.** Workflow `evaluate-code`, with `branch`, the full `commit`, `tier` and
   `verdict` in the header. The guard hook reads it at `gh pr create`.

**Re-check after fixes.** Fixes are applied only with approval (**gate**) and committed.
Then: rerun step 3; verify each fixed finding against `git diff <old-commit>..HEAD`;
rerun the panel only if the fix diff touches a high-tier path beyond the lines of the
findings. Write a new report for the new commit.

## Mode: results

Question: can these numbers be reported? Run folders, logs and `sacct` are read from
`$MAIN`.

1. **Inventory.** Write down the expected runs (arms × folds × seeds) before looking.
   Then:
   ```bash
   ls -d "$MAIN"/checkpoints/<run-prefix>-fold*-seed* | wc -l
   for d in "$MAIN"/checkpoints/<run-prefix>-fold*-seed*; do [ -f "$d/results.json" ] || echo "no results: $d"; done
   sacct -u "$USER" -S <start-date> --format=JobID,JobName%45,State,ExitCode,Elapsed -P | grep -v "\.batch\|\.extern"
   ```
   A missing or failed run goes to `/sfx-debug`; it is not silently left out.
2. **Integrity.** For each run: the config, the seed and the commit it ran with; that it
   finished all epochs; no requeue or override anomaly (`override_applied.json`,
   `wandb_id.txt`, restart count in the log). `results.json` holds only `fold_id`,
   `test_detector`, `in_domain`, `cross` and `inference_threshold`, so config, seed and
   commit come from the W&B run and the SLURM log. Where one cannot be established,
   write "not established". Do not assume.
3. **Aggregate.**
   ```bash
   cd "$MAIN" && python scripts/aggregate_lodo_results.py --select all
   ```
   Seeds are reported separately by default; `--pool-seeds` averages them per fold.
   Report per fold and pooled, with n.
4. **Compare arms.** For each pair: the difference in the metric, next to the spread
   across seeds within each arm (min–max, n stated). A difference smaller than the
   larger of the two seed ranges is written as "inside the seed spread". With n = 3
   seeds this is a description, not a significance test; say so.
5. **Validity.** Run the `validity` mode below before any verdict.
6. **Verdict and report.** `trust`, `trust-with-caveats` (caveats listed) or
   `do-not-report`. Workflow `evaluate-results`. **Gate** on any edit to decks, notes,
   `PLANNING.md`, the kanban file or memory. Deck updates use the `progress-report`
   skill: ask the user to type `/progress-report` (Claude cannot invoke it).

## Mode: validity

Question: should this experiment be trusted? Runs inside `results` or on its own.

Give the `sfx-ml-reviewer` agent the runs under audit and this list. It uses categories
3 and 4 of `ml-review-guidelines.md` and the two calibration-leakage rules.

| Check | Evidence to gather |
|---|---|
| Splits | No beamtime session in both training and evaluation data (`build_sessions` in `src/training/lodo.py`; split files under `checkpoints/asymmetric_splits/` and `checkpoints/intra_splits/`). Each pretrain checkpoint excludes its held-out detector |
| Seeds | Each seed changes what it claims to change: model init, crop sampling, data order. Two seeds of one arm do not produce identical numbers |
| Arms | The configs of two arms differ only in the intended variable: `diff` the resolved configs |
| Preprocessing | Identical across tracks; the frame cache was built with the parameters the run used (cache manifest against the run config) |
| Evaluation protocol | Vote aggregation; the threshold was chosen on validation data and frozen (`val_threshold` in `best.pt`, `inference_threshold` in `results.json`); the same evaluation set for every arm |
| Code version | The same commit for every arm, or the differences are listed and shown not to matter |
| Calibration provenance | How the production CXI files were calibrated, and which frames the PF8 hitfinder parameters were tuned on. Recorded once per dataset in `$MAIN/reports/workflows/provenance.md`; read it first, and ask the user only for what it does not answer |

Each check ends as `holds`, `fails` or `not established`, with its evidence. Verdict:
`trust`, `trust-with-caveats` or `do-not-report`. A check that is `not established` is
a caveat at least. Workflow `evaluate-validity` when run on its own.

## Hard rules

- Report confirmed findings only, and numbers with n.
- "Not established" is an answer. A guess is not.
- A verdict of `ready` or `trust` is never given to save time.
