---
name: slurm-job
description: Submit a SLURM training job for sfx-hitfinder by copying the canonical template (grp_cxfel QOS, H100 on scg020) with wandb env and source activate
disable-model-invocation: true
dashboard-confirm: true
dashboard-art: rocket
---

Get these four values. If you were started by the dashboard or another agent, nobody can answer questions: read them from the task's "Arguments:" line (free text, for example `configs/supervised/resnet18_resonet.yaml resnet18-resonet-fold0 04:00:00 128G`; config path first, then job name, then optional time and memory). If the config path or job name is missing, stop and report that instead of guessing, because this submits a real GPU job. Otherwise ask the user for:
1. Config file path (e.g. configs/supervised/resnet18_resonet.yaml)
2. Job name (e.g. resnet18-resonet-fold0)
3. Time limit (default: 02:00:00, the template's value)
4. Memory (default: 128G, the template's value)

The GPU is fixed by the template (one H100 on scg020 under QOS grp_cxfel), so do not ask for a GPU count.

Copy `scripts/templates/slurm_gpu_job.sh` to `scripts/submit_<job_name>.sh` and change only these lines. Do not edit the template itself.

- `#SBATCH --job-name=sfx-<job_name>`
- `#SBATCH --time=<time_limit>`
- `#SBATCH --mem=<memory>`
- `#SBATCH --output=logs/<job_name>-%j.out` and `#SBATCH --error=logs/<job_name>-%j.err`
- the last line, replaced by:

```bash
python -m src.training.train_asymmetric --config <config_path>
```

Leave `-p general`, `-q grp_cxfel`, `--gres=gpu:h100:1`, `--nodelist=scg020`, `source activate sfx-hitfinder` and `source .secrets/wandb.env` exactly as they are. Never put a wandb key in the script.

Then run:
```bash
mkdir -p logs
sbatch scripts/submit_<job_name>.sh
squeue -u $USER
```

Remind the user to run `/watch-slurm` to monitor the job, and `/slurm-log-analyzer` if the job fails.
