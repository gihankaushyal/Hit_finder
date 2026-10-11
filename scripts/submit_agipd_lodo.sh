#!/bin/bash
#SBATCH --job-name=sfx-agipd-lodo
#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --output=logs/agipd-lodo-%j.out
#SBATCH --error=logs/agipd-lodo-%j.err

# DEPRECATED (pipeline v1): this script writes the legacy run prefix
# 'legacy-asymmetric-v1', which predates the frame cache and the run-naming standard.
# It refuses to run so a v1 run cannot be started by accident. The #SBATCH header
# asks for no GPU or node, so the refusal job starts at once and its notice
# appears in the .err log (sbatch itself still prints 'Submitted batch job').
echo "DEPRECATED: $(basename "$0") is a pipeline-v1 script (run prefix 'legacy-asymmetric-v1') and no longer runs." >&2
echo "Use instead:" >&2
echo '  bash scripts/submit_asymmetric_lodo_all.sh --run-name-prefix resnet18-asymmetric-v2 --folds 1' >&2
echo "See src/training/run_naming.py for the run-name convention." >&2
exit 1


module load mamba/latest
source activate sfx-hitfinder

source .secrets/wandb.env

python -u -m src.training.train_asymmetric \
    --config configs/supervised/resnet18_asymmetric.yaml \
    --run-name-prefix legacy-asymmetric-v1 \
    --folds 1 \
    --tags supervised,resnet18,asymmetric-pipeline,agipd-lodo-rerun
