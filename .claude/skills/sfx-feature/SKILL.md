---
name: sfx-feature
description: Feature development workflow for the Hit_finder repo. Use when the user asks to add, change or remove behaviour in src/, scripts/, configs/ or SLURM scripts ("implement", "add", "build", "change how X works"), or when /sfx-debug hands over a fix. Runs a full tier (design, plan, worktree, review, PR) or a fast tier (small change on the phase branch) and stops at every gate.
---

# sfx-feature

Say first: "Starting the sfx-feature workflow." If you started it yourself, without the
slash command, say that too.

## Start

1. Read `.claude/skills/sfx-shared/gates.md` and `.claude/skills/sfx-shared/invariants.md`.
2. Find the main checkout and the phase branch. Shell variables do not survive between
   Bash calls, so repeat these two lines at the start of any command that uses them:
   ```bash
   MAIN="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
   PHASE="$(git -C "$MAIN" branch --show-current)"
   ```
3. Pick the tier and say it in one line with the reason. The user can change it.

| Tier | When |
|---|---|
| Full | Behaviour changes under `src/`; a new training or evaluation script; anything touching preprocessing, splits, seeds or metrics |
| Fast | Docs, configs that only add a run, small script fixes, tests only |

Unsure → full.

## Full tier

1. **Explore.** `graphify query "<question>"` first (see the Knowledge Graph section of
   `CLAUDE.md` for which graph). Use the `feature-dev:code-explorer` agent only when the
   change spans several modules.
2. **Design.** Invoke `superpowers:brainstorming`. Ask with AskUserQuestion until the
   requirements are clear (≥95%). The spec goes to `docs/superpowers/specs/` (local,
   never committed). **Gate:** the user approves the design.
3. **Plan.** Invoke `superpowers:writing-plans`. The plan goes to
   `docs/superpowers/plans/`. **Gate:** the user approves the plan.
4. **Worktree.** From the main checkout, without switching its branch:
   ```bash
   cd "$MAIN" && git worktree add "../Hit_finder-wt/$PHASE-<name>" -b "$PHASE-<name>" "$PHASE"
   cd "../Hit_finder-wt/$PHASE-<name>"
   ln -s "$MAIN/.secrets" .secrets
   ln -s "$MAIN/src/hitfinders/_pf8_wrap.so" src/hitfinders/_pf8_wrap.so
   ```
   `<name>` is lowercase kebab-case. `data/` is tracked as empty folders and needs no
   link; if a `data/` sub-folder in the main checkout is a symlink, recreate the same
   link. If the feature changes `_pf8_wrap.c`, build instead of linking:
   `make -C src/hitfinders`. Run `npm install` in `dashboard/` only if the feature
   touches it.
5. **Implement.** Invoke `superpowers:executing-plans`. Write the failing test first.
   The black hook formats each edited file. If the change touches
   `src/preprocessing/`: read `.claude/skills/pipeline-check/SKILL.md` and run its
   commands (Claude cannot invoke that skill directly). Any change to normalisation or
   feature scaling carries a note with file:line, tensor shape and dtype before and
   after.
6. **Commit. Gate.** The review report names a commit, so commit before the review.
7. **Review.** Run `/sfx-evaluate code`. Also run `/sfx-security diff` if a shell
   script, a SLURM script, secrets handling, a hook or a dependency file changed. Fix
   what the user approves (**gate** per fix), commit, and let `/sfx-evaluate code` do its
   re-check until the verdict is `ready`.
8. **Ship.** **Gate:** push. **Gate:** open the PR into `$PHASE` (never `main`):
   ```bash
   git push -u origin "$PHASE-<name>"
   gh pr create --base "$PHASE" --title "<title>" --body "<body>"
   ```
   The guard hook asks at `gh pr create` unless the newest evaluate-code report for
   this branch and commit says `ready`. Then run `/code-review high`: invoke the
   `code-review` skill with `high` if it is available to you, otherwise ask the user to
   type it. Wait for CI. **Gate:** merge.
9. **After merge.**
   ```bash
   cd "$MAIN" && git worktree remove "../Hit_finder-wt/$PHASE-<name>"
   squeue -u "$USER" -h -o "%i %j %T"
   ```
   Pull the main checkout only when no queued or running job depends on the changed
   files. **Gate:** `git pull`. **Gate:** any update to the kanban file or `PLANNING.md`.
10. **Report.** Write the report (`.claude/skills/sfx-shared/report-template.md`,
    workflow `feature`): what was built, the PR, each gate and its answer.

## Fast tier

1. A design note of three to six lines: what changes, which files, how it is checked.
   **Gate.**
2. `squeue -u "$USER" -h -o "%i %j %T"` — if a queued or running job reads a file you
   are about to change, say so and wait for the user's decision.
3. Edit on `$PHASE` in the main checkout. Never switch its branch.
4. Run the tests that cover the change, and `black --check` on changed Python files.
   Review inline, or with one reviewer if `.claude/skills/sfx-shared/risk-tiers.md` puts
   the change at medium. A change that turns out to be high tier moves to the full tier.
5. **Gate:** commit. **Gate:** push.
6. Report, as in the full tier (verdict `merged` once pushed to `$PHASE`).

## Hard rules

- No significant feature without an approved design and an approved plan.
- Never switch branches in the main checkout.
- A preprocessing change applies to both tracks and needs a design review (invariant I7).
- `docs/superpowers/` stays local. Never `git add -f`.
