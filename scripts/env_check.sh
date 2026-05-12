#!/bin/bash
# Run on a compute node: srun --partition=htc --gpus=1 --pty bash scripts/env_check.sh

module load mamba/latest
conda activate sfx-hitfinder

python -c "
import torch, h5py, timm, wandb
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('h5py:', h5py.__version__)
print('timm:', timm.__version__)
print('wandb:', wandb.__version__)
try:
    import reborn
    print('reborn: OK')
except ImportError:
    print('reborn: NOT FOUND')
"
