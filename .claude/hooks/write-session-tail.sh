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
mv "$TMP" "$TAIL_FILE"
exit 0
