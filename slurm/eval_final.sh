#!/bin/bash
# CPU: metrics (scripts/evaluate.py, unchanged) for every finished final run.
#   sbatch slurm/eval_final.sh          # 12 tasks: 0-3 SteerViT, 4-7 CLIP, 8-11 SAM3
#SBATCH --job-name=medsteer-evalfinal2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
OUT=${OUT:-runs_final2}
RUNS=($OUT/gate1 ${OUT}_compare/clip/gate1 ${OUT}_compare/sam3/gate1)
DATASETS=(brisc busi kvasir isic)
R=${RUNS[$((SLURM_ARRAY_TASK_ID / 4))]}
D=${DATASETS[$((SLURM_ARRAY_TASK_ID % 4))]}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.3 2>/dev/null || true
source .venv/bin/activate
python scripts/evaluate.py --run "$R/$D" --manifest "${MANIFEST:-data/manifest_final.csv}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
