# Workflow report format

Every workflow ends with a terminal summary and a report file.

**Where:** `<MAIN>/reports/workflows/<YYYY-MM-DD>-<workflow>-<topic>.md`, always in the
main checkout, also when the session runs in a worktree:

```bash
MAIN="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
mkdir -p "$MAIN/reports/workflows"
```

`<workflow>` is one of `feature`, `debug`, `security-diff`, `security-full`,
`evaluate-code`, `evaluate-results`, `evaluate-validity`. `<topic>` is kebab-case, five
words at most. `reports/` is gitignored: never `git add -f` a report.

## Header

The file starts with plain `key: value` lines between two `---` lines. The guard hook
reads this header from `evaluate-code` reports (first 20 lines), so spell the keys
exactly, lower-case, one per line, no markdown.

```
---
workflow: evaluate-code
date: 2026-11-11
branch: phase-05-example
commit: 0123456789abcdef0123456789abcdef01234567
tier: high
verdict: ready
---
```

- `commit` is the full output of `git rev-parse HEAD`. Workflows not tied to a commit
  (`debug`, `evaluate-results`, `evaluate-validity`, `security-full`) still record the
  commit of the checkout they read.
- `tier` is `high`, `medium`, `low`, or `n/a`.
- `verdict`:

| Workflow | Values |
|---|---|
| `evaluate-code` | `ready`, `ready-after-fixes`, `not-ready` |
| `evaluate-results`, `evaluate-validity` | `trust`, `trust-with-caveats`, `do-not-report` |
| `security-diff`, `security-full` | `clean`, `findings` |
| `debug` | `root-cause-confirmed`, `open` |
| `feature` | `merged`, `pr-open`, `stopped` |

Only `verdict: ready` on an `evaluate-code` report for the current branch and commit
lets `gh pr create` run without a prompt.

## Body

```markdown
# <Workflow>: <topic>

## Summary
Three lines at most: what was checked, the verdict, the one thing to do next.

## What ran
Commands and agents, each with its result (pass, fail, counts). Say what was skipped
and why.

## Findings
By severity (🔴, 🟡, 🟢). Each: file:line, evidence, proposed fix, and whether it was
verified. Findings that could not be confirmed go under "Unverified, dropped".

## Design notes
Conflicts between a review guideline and the fixed architecture. For discussion; they
never change the verdict.

## Pre-existing gaps
Problems the change did not cause. Listed once; they never block.

## Gates
Each gate reached, what was asked, and the answer.

## Next
What the user decides now.
```

A section with nothing to say is written as "None." and kept.

## Rules

- Redact any secret value: write `<redacted>`, the file and the line number. Never the
  value, in the report or in the terminal.
- State what is not known. "Not established" is a valid entry; a guess is not.
- Numbers come with n (folds, seeds, frames).
