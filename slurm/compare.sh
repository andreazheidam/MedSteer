#!/bin/bash
# CLIP + SAM3 comparison maps: one GPU job per (model, dataset), as a job array.
#   sbatch slurm/compare.sh                 # all 8 jobs
#   sbatch --array=0-3 slurm/compare.sh     # CLIP only (SAM3 needs Hugging Face access)
#   sbatch --array=4-7 slurm/compare.sh     # SAM3 only
# Resumable: finished cases are skipped, so just resubmit after a timeout.
#SBATCH --job-name=medsteer-compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
MODELS=(clip sam3)
DATASETS=(brisc busi kvasir sunnybrook)
M=${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}
D=${DATASETS[$((SLURM_ARRAY_TASK_ID % 4))]}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.3 2>/dev/null || true
source .venv-compare/bin/activate          # transformers >= 5 (SAM3); NOT the SteerViT env
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}
SPLIT=""; [ "$D" = "brisc" ] && SPLIT="--splits test"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "model $M  dataset $D"
python scripts/extract_compare.py --model "$M" --dataset "$D" --config configs/prompts.yaml \
    --manifest "${MANIFEST:-data/manifest_all.csv}" --out "${OUT:-runs_compare}" $SPLIT $([ "$M" = sam3 ] && echo --batch-size 1) $([ "$M" = sam3 ] && echo --batch-size 1)
