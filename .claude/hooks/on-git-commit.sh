#!/usr/bin/env bash
# PostToolUse hook: fires after every Bash tool call.
# Detects successful git commits and then:
#   1. Runs graphify update . (AST-only, no API cost)
#   2. Appends a compact commit note to the branch-mapped archive file
set -euo pipefail

INPUT=$(cat -)

# Only act on git commit commands
CMD=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('tool_input', {}).get('command', ''))
except Exception:
    print('')
" "$INPUT" 2>/dev/null || echo "")

case "$CMD" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

# Only act when commit succeeded (output starts with "[branchname hash]")
OUTPUT=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('tool_response', {}).get('output', ''))
except Exception:
    print('')
" "$INPUT" 2>/dev/null || echo "")

case "$OUTPUT" in
  "["*) ;;
  *) exit 0 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Memory lives under the main checkout's project id, also when this runs from a worktree.
MAIN_ROOT="$(git -C "$REPO" worktree list --porcelain 2>/dev/null | sed -n '1s/^worktree //p')"
MEM_DIR="$HOME/.claude/projects/$(printf '%s' "${MAIN_ROOT:-$REPO}" | sed 's|[^A-Za-z0-9]|-|g')/memory"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%MZ")
TODAY=$(date +"%Y-%m-%d")

cd "$REPO"

# 1. Update graphify knowledge graph (AST-only, no API cost)
[ -f graphify-out/graph.json ] && graphify update . >/dev/null 2>&1 || true

# 2. Gather commit metadata
COMMIT=$(git log --oneline -1 2>/dev/null || echo "unknown")
DIFF_STAT=$(git diff --stat HEAD~1..HEAD 2>/dev/null | tail -1 | sed 's/^ *//' || echo "")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

# 3. Map branch to archive file (mirrors /checkpoint skill mapping)
case "$BRANCH" in
  *ssl*|*phase-05*) ARCHIVE="project_phase5_ssl_progress.md" ;;
  *lodo*|*asymmetric*|*phase-04*) ARCHIVE="project_asymmetric_pipeline.md" ;;
  *gpu*|*hitfinder*) ARCHIVE="project_gpu_hitfinder.md" ;;
  *resonet*|*cxi*) ARCHIVE="project_resonet_cxi_integration.md" ;;
  *) ARCHIVE="project_phase_status.md" ;;
esac

TARGET="${MEM_DIR}/${ARCHIVE}"
[ -f "$TARGET" ] || exit 0

# 4. Append compact commit note
printf '\n## Commit note %s (branch: %s)\n**Commit:** %s\n**Summary:** %s\n' \
  "$TIMESTAMP" "$BRANCH" "$COMMIT" "$DIFF_STAT" >> "$TARGET"

# 5. Update last_verified frontmatter date
python3 -c "
import re, sys
path, today = sys.argv[1], sys.argv[2]
text = open(path).read()
updated = re.sub(r'last_verified: \d{4}-\d{2}-\d{2}', f'last_verified: {today}', text, count=1)
open(path, 'w').write(updated)
" "$TARGET" "$TODAY" 2>/dev/null || true

exit 0
