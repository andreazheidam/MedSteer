#!/bin/bash
# FINAL-2 experiment: frozen prompts (configs/prompts_final2.yaml), datasets BRISC / BUSI / Kvasir / ISIC.
# One GPU job per (model, dataset):  0-3 SteerViT, 4-7 CLIP, 8-11 SAM3 (order brisc busi kvasir isic).
#   sbatch slurm/final.sh                    # all 12
#   sbatch --array=0-3 slurm/final.sh        # SteerViT only
# Resumable: finished cases are skipped (so never point OUT at the v2 runs); resubmit the same task after a timeout.
#SBATCH --job-name=medsteer-final2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
MODELS=(steervit clip sam3)
DATASETS=(brisc busi kvasir isic)
M=${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}
D=${DATASETS[$((SLURM_ARRAY_TASK_ID % 4))]}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.3 2>/dev/null || true
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}
CONFIG=${CONFIG:-configs/prompts_final2.yaml}
OUT=${OUT:-runs_final2}            # NEW folders: extraction skips cases that already have an .npz
MANIFEST=${MANIFEST:-data/manifest_final.csv}
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "model $M  dataset $D  config $CONFIG ($(sha256sum $CONFIG | cut -c1-12))"

if [ "$M" = steervit ]; then
  source .venv/bin/activate
  python scripts/extract.py --config "$CONFIG" --manifest "$MANIFEST" --dataset "$D" \
      --out "$OUT" --attention
else
  source .venv-compare/bin/activate
  python -c "import torch, sys, socket; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible to torch on ' + socket.gethostname())"
  BS=""; [ "$M" = sam3 ] && BS="--batch-size 1"
  python scripts/extract_compare.py --model "$M" --dataset "$D" --config "$CONFIG" \
      --manifest "$MANIFEST" --out "${OUT}_compare" $BS
fi
