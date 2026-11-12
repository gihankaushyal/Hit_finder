---
name: sfx-debug
description: Debugging workflow for the Hit_finder repo. Use for a failed or timed-out SLURM job, a failing test or CI run, a result that looks wrong (a metric that dropped, jumped or is too good), or a data or preprocessing bug. Finds the root cause before any fix, lists the runs the bug affected, and stops at every gate.
---

# sfx-debug

Say first: "Starting the sfx-debug workflow." If you started it yourself, say that too.

Backbone: invoke `superpowers:systematic-debugging` and follow it. This file adds what
is specific to this repo.

## Start

1. Read `.claude/skills/sfx-shared/gates.md` and `.claude/skills/sfx-shared/invariants.md`.
2. Find the main checkout. Logs, checkpoints and SLURM state live there, also when the
   session runs in a worktree:
   ```bash
   MAIN="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
   ```

## 1. Triage

Say which kind it is. One problem can be two kinds; start with the first that applies.

**SLURM failure**
```bash
sacct -j <jobid> --format=JobID,JobName%40,State,ExitCode,Elapsed,Start,End,NodeList -P
ls -1t "$MAIN"/logs/*<jobid>* 2>/dev/null || ls -1t "$MAIN"/logs/*.err | head
```
Give the confirmed log paths to the `slurm-log-analyzer` agent. Check the restart
count and the markers in the run folder: `override_applied.json` (a requeued job must
not repeat `--override-training`) and `wandb_id.txt`. Find out which commit the job
read: jobs read the live main checkout, so compare the job's start time with
`git -C "$MAIN" reflog --date=iso | head`.

**Test or CI failure**
```bash
python -m pytest <file>::<test> -x -q
gh run list --branch "$(git branch --show-current)" --limit 5
gh run view <run-id> --log-failed | tail -60
```
Reproduce the single test before reading code. A test that fails only in CI: compare
`requirements-ci.txt` with the local environment (CPU-only, no compiled PF8).

**Suspicious result**
Compare with a known-good run: diff the two configs, the two commits and the two seeds.
Write the hypotheses down, ranked. One diagnostic per hypothesis, on real frames, each
with the outcome that would confirm it and the outcome that would rule it out. Check
the cheap explanations first: wrong checkpoint loaded, wrong split file, stale frame
cache, a requeue that restarted training, a seed that did not change what it claims to.

**Data or pipeline bug**
Read `.claude/skills/pipeline-check/SKILL.md` and run its commands. For geometry and
assembly questions query the reborn graph (`cd ../reborn && graphify query "..."`), for
CXI generation the Resonet graph; return to the project root afterwards.

Agents this workflow may run: `slurm-log-analyzer`; `sfx-ml-reviewer` (does this code
do what the experiment assumes?); `pr-review-toolkit:silent-failure-hunter` (errors
swallowed on the failing path).

## 2. Root cause

State it in this form:

> **Root cause:** <what is wrong, file:line>. **Evidence:** <what was observed>.
> **Why it produced the symptom:** <the chain>. **Not yet explained:** <or "nothing">.

If the evidence does not single out one cause, say which hypotheses remain and what
would separate them. **Gate:** the user accepts the root cause before any fix.

## 3. Fix

Write the regression test first and watch it fail for the right reason. Then hand over
to `/sfx-feature` (fast tier for a small fix, full tier if behaviour under `src/`
changes).

## 4. Blast radius

List the runs the bug affected: every run that used the faulty code path between the
commit that introduced it (`git log -S'<string>' --oneline -- <file>`) and the fix.
For each: run name, fold, seed, job id, and whether its numbers appear in a deck, note
or memory entry. **Gate** before any rerun, `sbatch`, or moving or deleting a
checkpoint. Archive, do not delete, unless the user says delete.

## 5. Report

Write the report (`.claude/skills/sfx-shared/report-template.md`, workflow `debug`,
verdict `root-cause-confirmed` or `open`): symptom, root cause, fix, blast radius,
gates. Propose a memory entry in the report (what to remember, why). **Gate:** writing
it to memory.

## Hard rules

- No fix before the root cause is accepted.
- No rerun, requeue, `scancel` or checkpoint move without a gate.
- A "fix" that makes the symptom disappear without explaining it is not a fix. Say so.
