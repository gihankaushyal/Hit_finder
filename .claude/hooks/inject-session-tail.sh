#!/usr/bin/env bash
# UserPromptSubmit hook: inject session-tail.md into every turn's context.
# Output is JSON with additionalContext — picked up by Claude Code before
# the user's message is processed. Total overhead: ~100 tokens per turn.
set -euo pipefail
cat - >/dev/null   # drain stdin

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAIL_FILE="${REPO}/session-tail.md"

CONTENT=$(cat "$TAIL_FILE" 2>/dev/null || echo "session-tail.md not found — run a turn to generate it")

python3 - "$CONTENT" <<'PYEOF'
import json, sys
content = sys.argv[1]
output = {
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": content
    }
}
print(json.dumps(output))
PYEOF
