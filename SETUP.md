# SETUP.md

Manual installation steps for dependencies not available via conda.

## Reborn

Install from the Kirianlab GitLab repository (gihan development branch):

```bash
pip install git+https://gitlab.com/kirianlab/reborn.git@gihan
```

For local development (editable install from a local clone):

```bash
pip install /path/to/reborn
```

Reborn uses a Meson/Fortran build system. On macOS, Apple Clang does not
include OpenMP — use Homebrew GCC instead:

```bash
CC=gcc-14 pip install git+https://gitlab.com/kirianlab/reborn.git@gihan
# or
CC=gcc-14 pip install /path/to/reborn
```

On Sol HPC (Linux), the standard install should work without CC override.

## SLURM Modules

```bash
module load mamba/latest
```

Check available modules on Sol: `module avail`
