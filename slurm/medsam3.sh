#!/bin/bash
# MedSAM3 extraction (SAM3 + medical LoRA), one GPU job per dataset: 0 brisc, 1 busi, 2 kvasir, 3 isic.
#   sbatch slurm/medsam3.sh              (needs MEDSAM3_ROOT and MEDSAM3_LORA, see below)
# Resumable: finished cases are skipped; resubmit the same task after a timeout.
#SBATCH --job-name=medsteer-medsam3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
DATASETS=(brisc busi kvasir isic)
D=${DATASETS[$SLURM_ARRAY_TASK_ID]}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}
MEDSAM3_ROOT=${MEDSAM3_ROOT:-$HOME/MedSAM3}
MEDSAM3_LORA=${MEDSAM3_LORA:?set MEDSAM3_LORA to the LoRA weights file}
source .venv-compare/bin/activate
python -c "import torch, sys, socket; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible to torch on ' + socket.gethostname())"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/extract_medsam3.py --medsam3-root "$MEDSAM3_ROOT" --lora "$MEDSAM3_LORA" \
    --dataset "$D" --config configs/prompts_final2.yaml --manifest "${MANIFEST:-data/manifest_final.csv}" \
    --out "${OUT:-runs_final2_compare}"
python scripts/evaluate.py --run "${OUT:-runs_final2_compare}/medsam3/gate1/$D" \
    --manifest "${MANIFEST:-data/manifest_final.csv}" --workers "${SLURM_CPUS_PER_TASK:-8}"
