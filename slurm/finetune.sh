#!/bin/bash
# Fine-tune SteerViT's steering modules (follow-up Sec. 2) and evaluate on the held-out test split.
# One run per line of $RUNS (default configs/finetune_kvasir.txt); the array index picks the line.
#   sbatch --array=0-2 slurm/finetune.sh          # frozen baseline, target-only, spec hard
#   sbatch --array=3-4 slurm/finetune.sh          # the two ablations
# Per task: train (train split, epoch chosen on val) -> extract the TEST split with the selected
# weights -> semdiff_eval. Afterwards, on the login node:
#   python scripts/semdiff_table.py --dataset kvasir --root runs_ft_eval --out paper_ft/kvasir \
#       --models steervit kvasir_target kvasir_spec_hard
#SBATCH --job-name=medsteer-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.3 2>/dev/null || true
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}

D=${DATASET:-kvasir}
RUNS=${RUNS:-configs/finetune_${D}.txt}
CONFIG=configs/prompts_semdiff.yaml
MANIFEST=${MANIFEST:-data/manifest_final.csv}
LINE=$(grep -v '^\s*#' "$RUNS" | grep -v '^\s*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$LINE" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $RUNS"; exit 1; }
NAME=$(echo "$LINE" | awk '{print $1}')
FT_ARGS=$(echo "$LINE" | cut -s -d' ' -f2- | xargs || true)
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "run $NAME  dataset $D  args: $FT_ARGS"

if [ "$NAME" = frozen ]; then
  python scripts/extract_semdiff.py --config $CONFIG --manifest $MANIFEST --dataset $D \
      --splits test --out runs_ft_eval --name steervit
  RUN=runs_ft_eval/steervit/gate1/$D
else
  python scripts/finetune.py --config $CONFIG --manifest $MANIFEST --dataset $D \
      --out runs_ft --run-name "$NAME" $FT_ARGS
  python scripts/extract_semdiff.py --config $CONFIG --manifest $MANIFEST --dataset $D \
      --splits test --out runs_ft_eval --name "$NAME" --weights "runs_ft/$NAME/ckpt_best.pt"
  RUN=runs_ft_eval/$NAME/gate1/$D
fi
python scripts/semdiff_eval.py --run $RUN --manifest $MANIFEST --workers "${SLURM_CPUS_PER_TASK:-8}"
