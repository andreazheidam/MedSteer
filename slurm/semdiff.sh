#!/bin/bash
# Training-free semantic differential steering (follow-up Sec. 1) on ONE dataset, five models:
#   0 steervit   1 clip   2 biomedclip   3 sam3   4 medsam3
#   sbatch slurm/semdiff.sh                                  # all five models, DATASET=kvasir
#   sbatch --array=0 slurm/semdiff.sh                        # SteerViT only (run this first)
#   sbatch --export=ALL,DATASET=brisc slurm/semdiff.sh       # another dataset
# MedSAM3 needs MEDSAM3_LORA (and MEDSAM3_ROOT, default ~/MedSAM3) as in slurm/medsam3.sh.
# Resumable: finished cases are skipped. After all tasks: python scripts/semdiff_table.py --dataset $DATASET
#SBATCH --job-name=medsteer-semdiff
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
MODELS=(steervit clip biomedclip sam3 medsam3)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
D=${DATASET:-kvasir}
CONFIG=configs/prompts_semdiff.yaml
OUT=${OUT:-runs_semdiff}
BASE=${MANIFEST:-data/manifest_final.csv}     # BRISC/ISIC: official test split; BUSI/Kvasir: all images
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data
module load python/3.11.3 2>/dev/null || true
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}

# lesion images only (non-empty masks), same rows for every model
LES=data/manifest_semdiff_${D}.csv
python - "$BASE" "$LES" "$D" <<'EOF'
import sys, csv
src, dst, ds = sys.argv[1:]
rows = [r for r in csv.DictReader(open(src)) if r["dataset"] == ds and r["mask_paths"]
        and r["label"] not in ("normal", "no_tumor")]
import os
tmp = f"{dst}.{os.getpid()}.tmp"       # one temp file per task: array tasks start at the same time
with open(tmp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
os.replace(tmp, dst)
print(f"{ds}: {len(rows)} lesion images -> {dst}")
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "model $M  dataset $D  config $CONFIG ($(sha256sum $CONFIG | cut -c1-12))"

RUN=$OUT/$M/gate1/$D
case $M in
  steervit)
    source .venv/bin/activate
    python scripts/extract_semdiff.py --config $CONFIG --manifest $LES --dataset $D --out $OUT ;;
  clip|biomedclip|sam3)
    source .venv-compare/bin/activate
    BS=""; [ "$M" = sam3 ] && BS="--batch-size 1"
    python scripts/extract_compare.py --model $M --dataset $D --config $CONFIG --manifest $LES --out $OUT $BS ;;
  medsam3)
    source .venv-compare/bin/activate
    python scripts/extract_medsam3.py --medsam3-root "${MEDSAM3_ROOT:-$HOME/MedSAM3}" \
        --lora "${MEDSAM3_LORA:?set MEDSAM3_LORA to the LoRA weights file}" \
        --dataset $D --config $CONFIG --manifest $LES --out $OUT ;;
esac
python scripts/semdiff_eval.py --run $RUN --manifest $LES --workers "${SLURM_CPUS_PER_TASK:-8}"
