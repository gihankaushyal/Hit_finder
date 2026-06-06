#!/usr/bin/env bash
# Setup and activate the sfx-hitfinder environment.
# USAGE: source setup_env.sh   (do NOT execute directly — must be sourced for conda activate to work)

# NOTE: do NOT use set -euo pipefail here — this script is sourced, so any unhandled
# error would exit the user's shell and close their terminal.

# Resolve project root relative to this script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Load mamba module if on Sol HPC
if command -v module &>/dev/null; then
    module load mamba/latest
fi

# Initialize conda shell functions (required for 'conda activate' to work)
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# 2. Create env from environment.yml if it doesn't exist
if conda env list | grep -q "^sfx-hitfinder[[:space:]]"; then
    echo "[setup_env] sfx-hitfinder environment already exists — skipping creation."
else
    echo "[setup_env] Creating sfx-hitfinder environment (this may take several minutes)..."
    echo "[setup_env] TIP: run this inside tmux in case of SSH disconnect."
    if mamba env create -f "$PROJECT_ROOT/environment.yml" -n sfx-hitfinder; then
        echo "[setup_env] Environment created successfully."
    else
        echo "[setup_env] ERROR: conda env create failed. Check the output above for details."
        echo "[setup_env] Common causes: package conflicts, network issues fetching reborn from GitLab."
        return 1
    fi
fi

# 3. Activate
if ! conda activate sfx-hitfinder; then
    echo "[setup_env] ERROR: conda activate failed. Try: conda init bash, then open a new terminal."
    return 1
fi

# 4. Add project root to PYTHONPATH for interactive sessions
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[setup_env] Done. Active env: $(conda info --envs | grep '\*' | awk '{print $1}')"
echo "[setup_env] Project root on PYTHONPATH: $PROJECT_ROOT"
