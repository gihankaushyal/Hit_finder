#!/usr/bin/env bash
# SessionStart hook — persist the sfx-hitfinder conda env across every Bash tool call.
#
# WHY: each Claude Code Bash tool call starts a fresh shell, so `conda activate`
# in one call does NOT carry to the next. Historically the agent re-typed
# `module load ... && conda activate sfx-hitfinder && <cmd>` on essentially every
# command (60+ full-path python invocations, 40+ module loads, 11 "command not
# found" errors across the session history). This hook writes the activation to
# $CLAUDE_ENV_FILE, which Claude Code sources before every Bash call, so bare
# `python`/`pytest`/`black`/`wandb` just work.
#
# SAFETY: no-op if CLAUDE_ENV_FILE is unset (older Claude Code) or the env is
# missing. Never exits non-zero. Does not touch ~/.bashrc or global state.
set +e
[ -z "${CLAUDE_ENV_FILE:-}" ] && exit 0

ENV_BIN="$HOME/.conda/envs/sfx-hitfinder/bin"
[ -x "$ENV_BIN/python" ] || exit 0

{
  # Best-effort full activation (sets CONDA_PREFIX, etc.) — silenced, never fatal.
  echo 'module load mamba/latest >/dev/null 2>&1 || true'
  echo 'if command -v conda >/dev/null 2>&1; then'
  echo '  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" >/dev/null 2>&1 || true'
  echo '  conda activate sfx-hitfinder >/dev/null 2>&1 || true'
  echo 'fi'
  # Guaranteed fallback: prepend env bin so python/pytest/black/wandb resolve
  # even if module/conda are unavailable in the tool shell.
  echo "export PATH=\"$ENV_BIN:\$PATH\""
} >> "$CLAUDE_ENV_FILE"

exit 0
