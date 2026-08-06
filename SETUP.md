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

## PF8 C Bridge (required for `hitfinder.backend: pf8`)

The CrystFEL PF8 ctypes backend requires a compiled shared library.
Run this once after cloning or after any change to `_pf8_wrap.c`:

```bash
module load gcc          # on Sol HPC — load the right GCC module
cd src/hitfinders
make
# Produces: src/hitfinders/_pf8_wrap.so
```

Verify the build:

```bash
python -c "from src.hitfinders.pf8 import PF8Hitfinder; print('OK')"
```

If `libcrystfel.so` is not found at runtime, add the library to the dynamic linker path:

```bash
export LD_LIBRARY_PATH=/data/bioxfel/software/crystfel-0.12.0/lib64:$LD_LIBRARY_PATH
```

Add this export to your `.bashrc` or SLURM job script if you see
`libcrystfel.so: cannot open shared object file`.

GSL version mismatch (`libgsl.so.27` not found): CrystFEL 0.12.0 was built
against GSL 2.7; if conda has a newer version, create a compatibility symlink:

```bash
ln -sf $CONDA_PREFIX/lib/libgsl.so.28.0.0 $CONDA_PREFIX/lib/libgsl.so.27
```
