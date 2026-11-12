# Risk tiers

The tier of a change is the highest tier of any file it touches.

List the files: `git diff --name-only "$PHASE"...HEAD`

| Tier | Paths | Review |
|---|---|---|
| High | `src/preprocessing/`, `src/data/`, `src/training/`, `src/evaluation/`; split or seed logic anywhere | `sfx-ml-reviewer`, `pr-review-toolkit:silent-failure-hunter`, `pr-review-toolkit:pr-test-analyzer` |
| Medium | other `src/` (`src/models/`, `src/hitfinders/`, `src/utils/`); `scripts/`; SLURM scripts; `.claude/hooks/`; `.claude/settings.json`; `.github/workflows/` | One reviewer: `sfx-ml-reviewer` for `src/`, otherwise `pr-review-toolkit:silent-failure-hunter`. Add `sfx-security-reviewer` if a shell script, a SLURM script, a hook, `settings.json` or a workflow file changed |
| Low | docs, `configs/`, `tests/`, `.claude/skills/`, `.claude/agents/` only | Inline, no agents |

Rules:

- Split or seed logic is high wherever it lives. `grep -nE "seed|split|fold" <diff>` on a
  medium or low change, and read each hit before settling the tier.
- A config change that alters an experiment's meaning (a preprocessing parameter, a
  split, a metric) is reviewed as high. A config change that only adds a run is low.
- Say the tier and the reason in one line before running anything. The user can raise or
  lower it.
- Run the agents the tier lists, in parallel, and no others. Starting a workflow is the
  user's request to run them.
- The generic code-reviewer agent is not in this panel: `/code-review high` runs on every
  PR, so a generic pre-PR review would pay twice for the same check.

Each agent gets: the worktree path, the exact diff range (`<base-sha>..<head-sha>`), the
tier, and the instruction to report findings caused by that diff only.
