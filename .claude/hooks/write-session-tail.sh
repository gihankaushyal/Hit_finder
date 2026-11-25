#!/usr/bin/env bash
# Stop hook: write session-tail.md after every Claude turn.
# Pure shell, no LLM, atomic write. ~80ms.
set -euo pipefail
cat - >/dev/null   # drain stdin to avoid SIGPIPE on hook JSON

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAIL_FILE="${REPO}/session-tail.md"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

BRANCH=$(git -C "$REPO" branch --show-current 2>/dev/null || echo "unknown")
COMMIT=$(git -C "$REPO" log --oneline -1 2>/dev/null || echo "unknown")
SLURM=$(squeue -u "${USER:-$(id -un)}" -h -o "%i %j %T" 2>/dev/null \
        | head -5 | tr '\n' ' | ' | sed 's/ | $//' || echo "none/unavailable")

NEXT_FILE="${REPO}/.session-next"
NEXT=$(head -1 "$NEXT_FILE" 2>/dev/null || echo "(not set — run /checkpoint to record)")

# A failed or unfinished doc backup (backup-docs.sh) leaves a marker next to memory/.
MAIN_ROOT=$(git -C "$REPO" worktree list --porcelain 2>/dev/null | sed -n '1s/^worktree //p' || true)
BACKUP_MARKER="${HOME:-}/.claude/projects/$(printf '%s' "${MAIN_ROOT:-$REPO}" | sed 's|[^A-Za-z0-9]|-|g')/doc-backup.FAILED"
BACKUP=""
if [ -f "$BACKUP_MARKER" ]; then
  BACKUP_LOG="${BACKUP_MARKER%.FAILED}.log"
  BACKUP="backup: FAILED $(head -1 "$BACKUP_MARKER" 2>/dev/null | cut -c1-200 || true) (log: ${BACKUP_LOG/#"${HOME:-}"/\~})"
fi

TMP=$(mktemp "${TAIL_FILE}.XXXXXX")
cat >"$TMP" <<EOF
# Session Tail — SFX Hitfinder
<!-- Auto-written by Stop hook after every Claude turn. Do not edit manually. -->
updated: ${TIMESTAMP}
branch: ${BRANCH}
commit: ${COMMIT}
slurm: ${SLURM:-none}
next: ${NEXT}
EOF
if [ -n "$BACKUP" ]; then
  printf '%s\n' "$BACKUP" >>"$TMP"
fi
mv "$TMP" "$TAIL_FILE"
exit 0
