#!/bin/bash
# Feature extraction for one dataset on one GPU (finished images are skipped, so resubmitting
# after a timeout continues where it stopped).
#   sbatch slurm/extract.sh kvasir
#   sbatch slurm/extract.sh brisc --gate 0.5          # extra args go to extract.py
# Check the GPU partition name on Hendrix with `sinfo` and adjust --partition / --gres.
#SBATCH --job-name=medsteer-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
DATASET=${1:?usage: sbatch slurm/extract.sh <brisc|busi|kvasir|sunnybrook> [extra args]}
shift || true
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.3 2>/dev/null || true   # Hendrix: same Python as the venv
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}   # model weights are cached here
MANIFEST=${MANIFEST:-data/manifest_all.csv}
OUT=${OUT:-runs}
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/extract.py --config configs/prompts.yaml --manifest "$MANIFEST" \
    --dataset "$DATASET" --out "$OUT" --attention "$@"
