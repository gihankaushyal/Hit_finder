---
name: sfx-security
description: Security review workflow for the Hit_finder repo (a public repository). Use with "diff" before a PR that touches shell or SLURM scripts, secrets handling, hooks, settings or dependencies; use with "full" on demand and at phase close for secrets in the repo and its history, the Claude setup, and Python and dashboard dependencies. Read-only; reports findings and never applies a fix without approval.
---

# sfx-security

Say first: "Starting the sfx-security workflow (<diff|full>)." If you started it
yourself, say that too. No argument means `diff`.

Read-only by default. This workflow never writes to `.secrets/` and never prints a
secret value: it reports the file, the line number and the kind of secret.

## Start

1. Read `.claude/skills/sfx-shared/gates.md`.
2. ```bash
   MAIN="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
   PHASE="$(git -C "$MAIN" branch --show-current)"
   PAT="(api[_-]?key|token|secret|passw(or)?d)[\"' ]*[:=][\"' ]*[A-Za-z0-9_\-]{16,}|\b[a-f0-9]{40}\b|wandb_v1_|ghp_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{20,}|-----BEGIN"
   ```
   Repeat these lines at the start of any command that uses the variables.

## 1. Secrets

`diff`:
```bash
git diff "$PHASE"...HEAD | grep -nEi "^\+.*($PAT)" | cut -d: -f1 | head
```
This prints line numbers of the diff output only. Open the diff at those lines to see
the file; do not echo the matching text.

`full`:
```bash
git grep -nEI -e "$PAT" -- . ':!dashboard/package-lock.json' | cut -d: -f1,2 | sort -u | head -50
git log --all --format='%h %ad %s' --date=short -G"wandb_v1_|ghp_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{20,}|-----BEGIN" | head -20
```
The first lists file:line in the working tree; the second lists commits whose diff
added or removed a key-like string. A 40-character hex string is usually a git commit
id: check the name on the same line before calling it a key.

Both modes:
```bash
git check-ignore -q .secrets/wandb.env && echo "ignored"
stat -c '%a %n' "$MAIN/.secrets" "$MAIN/.secrets/wandb.env"
grep -nE "^\s*(env|printenv|export -p)\s*$|set -x|echo .*WANDB_API_KEY" scripts/*.sh scripts/templates/*.sh
grep -cE "$PAT" "$MAIN/.claude/settings.local.json"
```
Expected: `ignored`; modes `700` and `600`; no script line that prints the environment
after the key is sourced; a count of `0` allow rules with a key-like string (if not 0,
report the count, never the rule text).

A key that is found in the repo or its history is reported for the user to **rotate**.
Removing it from a file does not un-leak it.

## 2. Code diff

Run the built-in security review (`security-review` skill if it is available to you,
otherwise ask the user to type `/security-review`), and the `sfx-security-reviewer`
agent with the checkout path and the range
`$(git merge-base "$PHASE" HEAD)..$(git rev-parse HEAD)`. In `full` mode with nothing
to diff, give the agent the stage, cleanup and submit scripts under `scripts/` and the
hook files under `.claude/hooks/` as the scope instead.

## 3. Claude setup audit (`full` only)

- **Allow rules.** In `$MAIN/.claude/settings.local.json` and `~/.claude/settings.json`:
  count the rules, and list those that allow a whole tool or a destructive prefix
  without limits (`Bash(*)`, `Bash(rm:*)`, `Bash(git push:*)`, `Bash(curl:*)`,
  `Bash(sudo:*)`), and any rule with an inline credential (count only).
- **Deny rules.** `permissions.deny` in `.claude/settings.json` still has
  `Read(./.secrets/**)`; the guard hook is still registered under `PreToolUse`.
- **Hooks.** For each hook in project and user settings: the event, the command, and
  whether the file it runs is tracked in this repo.
- **MCP servers.** `claude mcp list`: each server, whether it connects, and where its
  token is stored (a file path, never the token).
- **Plugins.** From `~/.claude/plugins/installed_plugins.json` and
  `~/.claude/plugins/known_marketplaces.json`: plugins that come from a marketplace
  other than the official one.

## 4. Dependencies (`full`, or `diff` when a dependency file changed)

Dependency files: `environment.yml`, `requirements-ci.txt`, `dashboard/package.json`,
`dashboard/package-lock.json`.

```bash
(cd "$MAIN/dashboard" && npm audit 2>&1 | tail -15)
command -v pip-audit >/dev/null && pip-audit 2>&1 | tail -20 || echo "pip-audit: not installed"
```
`pip-audit` with no arguments audits the active environment. If it is not installed:
**gate** on installing it; if the user declines, the report says "Python dependencies:
not audited".

## 5. Report

Write the report (`.claude/skills/sfx-shared/report-template.md`, workflow
`security-diff` or `security-full`, verdict `clean` or `findings`). Findings by
severity, each with evidence and a proposed fix. **Gate:** no fix is applied without
approval; an approved fix goes through `/sfx-feature`.

## Hard rules

- Never print, copy or store a secret value. Never write to `.secrets/`.
- Read-only: this workflow changes no file except its report.
- The repo is public. Anything committed stays public.
