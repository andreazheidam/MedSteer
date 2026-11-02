#!/bin/bash
# BiomedCLIP extraction + evaluation, one GPU job per dataset: 0 brisc, 1 busi, 2 kvasir, 3 isic.
#   sbatch slurm/biomedclip.sh
#SBATCH --job-name=medsteer-biomedclip
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
DATASETS=(brisc busi kvasir isic)
D=${DATASETS[$SLURM_ARRAY_TASK_ID]}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
source .venv-compare/bin/activate
python -c "import torch, sys, socket; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible to torch on ' + socket.gethostname())"
python scripts/extract_compare.py --model biomedclip --dataset "$D" --config configs/prompts_final2.yaml \
    --manifest "${MANIFEST:-data/manifest_final.csv}" --out "${OUT:-runs_final2_compare}"
python scripts/evaluate.py --run "${OUT:-runs_final2_compare}/biomedclip/gate1/$D" \
    --manifest "${MANIFEST:-data/manifest_final.csv}" --workers "${SLURM_CPUS_PER_TASK:-8}"
