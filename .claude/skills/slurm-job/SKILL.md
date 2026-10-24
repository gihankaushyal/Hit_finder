---
name: slurm-job
description: Submit a SLURM training job for sfx-hitfinder using the standard template with wandb env, conda activation, and A100 GPU config
disable-model-invocation: true
dashboard-confirm: true
dashboard-art: rocket
---

Ask the user for:
1. Config file path (e.g. configs/supervised/resnet18_resonet.yaml)
2. Job name (e.g. resnet18-resonet-fold0)
3. GPU count (default: 1)
4. Time limit (default: 12:00:00)
5. Memory (default: 64G)

Write the job script to `scripts/submit_<job_name>.sh` using this exact template:

```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --partition=htc
#SBATCH --gres=gpu:<gpu_count>
#SBATCH --cpus-per-task=8
#SBATCH --mem=<memory>
#SBATCH --time=<time_limit>
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

source .secrets/wandb.env

python scripts/train_asymmetric.py --config <config_path>
```

Then run:
```bash
sbatch scripts/submit_<job_name>.sh
squeue -u $USER
```

Remind the user to run `/watch-slurm` to monitor the job, and `/slurm-log-analyzer` if the job fails.
