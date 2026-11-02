#!/bin/bash
# Gate-strength sweep: one GPU job per (gate, dataset), submitted as a job array.
#   sbatch slurm/gate_sweep.sh
# Gate 1.0 is the released model (already extracted); 0 means no steering.
# Afterwards, per gate:  GATE=0.5 sbatch slurm/evaluate.sh <dataset>   then   GATE=0.5 sbatch slurm/evaluate.sh analyze
#SBATCH --job-name=medsteer-gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
GATES=(0.25 0.5 0.75 1.5)
DATASETS=(brisc busi kvasir sunnybrook)
G=${GATES[$((SLURM_ARRAY_TASK_ID / 4))]}
D=${DATASETS[$((SLURM_ARRAY_TASK_ID % 4))]}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.3 2>/dev/null || true   # Hendrix: same Python as the venv
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
SPLIT=""; [ "$D" = "brisc" ] && SPLIT="--splits test"
echo "gate $G  dataset $D"
python scripts/extract.py --config configs/prompts.yaml --manifest "${MANIFEST:-data/manifest_all.csv}" \
    --dataset "$D" --out "${OUT:-runs}" --gate "$G" --attention $SPLIT
