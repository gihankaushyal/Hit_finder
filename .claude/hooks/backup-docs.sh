#!/usr/bin/env bash
# Back up the local-only project docs into a git repository outside the checkout.
#
# Usage: backup-docs.sh [MAIN_CHECKOUT]
#   (default: the main checkout of the repository the script is run in)
#
# Copies PLANNING.md, MEMORY.md, the phase kanban file, docs/progress_notes.md,
# docs/superpowers/ and reports/ into
#   ~/.claude/projects/<project-id>/doc-backup/
# and commits there when something changed. session-tail.md and .session-next are
# regenerated every turn and are not backed up. A file deleted from the checkout leaves
# the backup's working tree but stays in its history.
#
# This script never fails its caller: every problem is written to stderr and the exit
# status is 0, so a broken backup can never block a git commit.
set -u

MAIN="${1:-$(git worktree list --porcelain 2>/dev/null | sed -n '1s/^worktree //p')}"
if [ -z "$MAIN" ] || [ ! -d "$MAIN" ]; then
  echo "backup-docs: no checkout found (${MAIN:-none})" >&2
  exit 0
fi

PROJECT_ID="$(printf '%s' "$MAIN" | sed 's|[^A-Za-z0-9]|-|g')"
DEST="$HOME/.claude/projects/$PROJECT_ID/doc-backup"

if ! mkdir -p "$DEST" 2>/dev/null; then
  echo "backup-docs: cannot create $DEST" >&2
  exit 0
fi
if [ ! -d "$DEST/.git" ] && ! git -C "$DEST" init -q 2>/dev/null; then
  echo "backup-docs: cannot initialise the backup repository in $DEST" >&2
  exit 0
fi

copy_file() { # path relative to the checkout
  [ -f "$MAIN/$1" ] || return 0
  mkdir -p "$DEST/$(dirname "$1")" && cp -p "$MAIN/$1" "$DEST/$1"
}

copy_dir() { # path relative to the checkout
  [ -d "$MAIN/$1" ] || return 0
  mkdir -p "$DEST/$1" || return 1
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$MAIN/$1/" "$DEST/$1/"
  else
    cp -a "$MAIN/$1/." "$DEST/$1/"
  fi
}

for f in PLANNING.md MEMORY.md docs/progress_notes.md; do
  copy_file "$f" || echo "backup-docs: could not copy $f" >&2
done
for k in "$MAIN"/phase-*-kanban.md; do
  [ -f "$k" ] && { copy_file "$(basename "$k")" || echo "backup-docs: could not copy $k" >&2; }
done
for d in docs/superpowers reports; do
  copy_dir "$d" || echo "backup-docs: could not copy $d/" >&2
done

git -C "$DEST" add -A >/dev/null 2>&1
if git -C "$DEST" diff --cached --quiet 2>/dev/null; then
  exit 0
fi
git -C "$DEST" -c user.name="doc-backup" -c user.email="doc-backup@localhost" \
  commit -q -m "backup $(date -u +%Y-%m-%dT%H:%MZ)" >/dev/null 2>&1 \
  || echo "backup-docs: commit failed in $DEST" >&2
exit 0
