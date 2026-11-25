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
# regenerated every turn and are not backed up. A file deleted from docs/superpowers/
# or reports/ leaves the backup's working tree but stays in its history; a deleted
# top-level file stays in the working tree too. Symlinks are copied as links.
#
# This script never fails its caller: every problem is written to stderr and the exit
# status is 0, so a broken backup can never block a git commit. So that a failure is
# still seen, ~/.claude/projects/<project-id>/doc-backup.FAILED is written when a run
# starts and removed only when it ends well; a failed or killed run leaves it behind,
# with the reason, and write-session-tail.sh shows it.
set -u

MAIN="${1:-$(git worktree list --porcelain 2>/dev/null | sed -n '1s/^worktree //p')}"
if [ -z "$MAIN" ] || [ ! -d "$MAIN" ]; then
  echo "backup-docs: no checkout found (${MAIN:-none})" >&2
  exit 0
fi

PROJECT_ID="$(printf '%s' "$MAIN" | sed 's|[^A-Za-z0-9]|-|g')"
STATE="$HOME/.claude/projects/$PROJECT_ID"
DEST="$STATE/doc-backup"
MARKER="$STATE/doc-backup.FAILED"
FAILED=""

fail() {
  echo "backup-docs: $*" >&2
  FAILED="${FAILED:+$FAILED; }$*"
}

finish() {
  if [ -n "$FAILED" ]; then
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$FAILED" >"$MARKER" 2>/dev/null
  else
    rm -f "$MARKER"
  fi
  exit 0
}

if ! mkdir -p "$DEST" 2>/dev/null; then
  echo "backup-docs: cannot create $DEST" >&2
  exit 0
fi
printf '%s run started, not finished\n' "$(date -u +%Y-%m-%dT%H:%MZ)" >"$MARKER" 2>/dev/null
if [ ! -d "$DEST/.git" ] && ! git -C "$DEST" init -q 2>/dev/null; then
  fail "cannot initialise the backup repository"
  finish
fi

copy_file() { # path relative to the checkout
  [ -f "$MAIN/$1" ] || [ -L "$MAIN/$1" ] || return 0
  mkdir -p "$DEST/$(dirname "$1")" && cp -P -p -- "$MAIN/$1" "$DEST/$1"
}

copy_dir() { # path relative to the checkout
  [ -d "$MAIN/$1" ] || return 0
  mkdir -p "$DEST/$1" || return 1
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete -- "$MAIN/$1/" "$DEST/$1/"
  else
    cp -a -- "$MAIN/$1/." "$DEST/$1/"
  fi
}

for f in PLANNING.md MEMORY.md docs/progress_notes.md; do
  copy_file "$f" || fail "could not copy $f"
done
for k in "$MAIN"/phase-*-kanban.md; do
  [ -f "$k" ] && { copy_file "$(basename "$k")" || fail "could not copy $k"; }
done
for d in docs/superpowers reports; do
  copy_dir "$d" || fail "could not copy $d/"
done

if ! ADD_ERR="$(git -C "$DEST" add -A 2>&1 >/dev/null)"; then
  ADD_ERR="${ADD_ERR//"$DEST"\//}" # git names the full path; the destination is fixed
  fail "git add failed: $(printf '%s\n' "$ADD_ERR" | head -1)"
  finish
fi
git -C "$DEST" diff --cached --quiet >/dev/null 2>&1
case $? in
  0) finish ;; # nothing changed
  1) ;;        # changes staged
  *)
    fail "backup repository unusable (git diff --cached failed)"
    finish
    ;;
esac
git -C "$DEST" -c user.name="doc-backup" -c user.email="doc-backup@localhost" \
  commit -q -m "backup $(date -u +%Y-%m-%dT%H:%MZ)" >/dev/null 2>&1 \
  || fail "commit failed"
finish
